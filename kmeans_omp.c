#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <omp.h>

struct KMData {
    int ndata;
    int dim;
    int** features;
    int* assigns;
    int* labels;
    int nlabels;
};

struct KMClust {
    int nclust;
    int dim;
    double** features;
    int* counts;
};

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines){
// Sets number of lines and total number of whitespace separated
// tokens in the file. Returns -1 if file can't be opened, 0 on
// success.
//
// EXAMPLE: int ret = filestats("digits_all_1e1.txt", &toks, &lines);
// toks  is now 7860 : 10 lines with 786 tokens per line, label + ":" + 28x28 pixels
// lines is now 10   : there are 10 lines in the file
  FILE *fin = fopen(filename,"r");
  if(fin == NULL){
    printf("Failed to open file '%s'\n",filename);
    return -1;
  }

  ssize_t ntokens=0, nlines=0, column=0;
  int intoken=0, token;
  while((token = fgetc(fin)) != EOF){
    if(token == '\n'){          // reached end of line
      column = 0;
      nlines++;
    }
    else{
      column++;
    }
    if(isspace(token) && intoken==1){ // token to whitespace
      intoken = 0;
    }
    else if(!isspace(token) && intoken==0){ // whitespace to token
      intoken = 1;
      ntokens++;
    }
  }
  if(column != 0){              // didn't end with a newline
    nlines++;                   // add a line on to the count
  }
  *tot_tokens = ntokens;
  *tot_lines = nlines;
  fclose(fin);
  // printf("DBG: tokens: %lu\n",ntokens);
  // printf("DBG: lines: %lu\n",nlines);
  return 0;
}


struct KMData * kmdata_load(struct KMData *data, char* datafile) {
    ssize_t tot_tokens, tot_lines;
    data->ndata = 0;
    int stat = filestats(datafile, &tot_tokens, &tot_lines);
    if (stat == -1) {
        printf("filestats return stat: %d", stat);
        return data;
    }

    FILE *fin = fopen(datafile, "r");
    if (fin == NULL) { printf("error opening file\n"); }
    data->ndata = tot_lines;
    int line_size = tot_tokens / tot_lines;
    int max_label = 0;
    data->labels = malloc(tot_lines * sizeof(int));
    data->features = malloc(tot_lines * sizeof(int*));
    for (int i=0; i<tot_lines; i++) {
        data->features[i] = malloc(line_size * sizeof(int));
    }
    char line[3142];
    int row = 0;
    int c;
    while (fgets(line, 3142*sizeof(char), fin) != NULL) {
        // printf("line= %s\n", line);
        char* token = strtok(line, " ");
        data->labels[row] = atoi(token);
        max_label = (atoi(token) > max_label) ? atoi(token) : max_label;
        token = strtok(NULL, " ");
        token = strtok(NULL, " ");
        c = 1;
        while (token != NULL) {
            data->features[row][c - 1] = atoi(token);
            token = strtok(NULL, " ");
            c++;
        }
        row++;
    }
    data->assigns = malloc(tot_lines * sizeof(int));
    data->dim = line_size - 2;
    data->nlabels = max_label + 1;
    fclose(fin);
    return data;
}

struct KMClust * kmclust_new(struct KMClust *clust, int nclust, int dim) {
    clust->nclust = nclust;
    clust->dim = dim;
    // clust->counts = malloc(nclust * sizeof(int));
    // (*clust->features)[nclust];
    // clust->features = malloc(sizeof(double[nclust][dim]));
    clust->features = malloc(nclust * sizeof(double*));
    clust->counts = malloc(nclust * sizeof(int));
    for (int i=0; i<nclust; i++) {
        clust->features[i] = malloc(dim * sizeof(double));
    }
    return clust;
}

void save_pgm_files(struct KMClust *clust, char* savedir) {
    int dim_root = (int)(sqrt((clust->dim)));
    if (clust->dim % dim_root == 0) {
        double maxfeats = 0.0;
        int dim = clust->dim;
        int nclust = clust->nclust;
        for (int i=0; i<nclust; i++) {
            for (int j=0; j<dim; j++) {
                maxfeats = (clust->features[i][j] > maxfeats) ? clust->features[i][j] : maxfeats;
            }
        }
        // for (int i=0; i<dim; i++) {
        //     for (int j=0; j<nclust; j++) {
        //         maxfeats = (clust->features[j][i] > maxfeats) ? clust->features[j][i] : maxfeats;
        //     }
        // }
        
        for (int c=0; c<nclust; c++) {
            char cent[10];
            char* pgm = ".pgm";
            char numbuf[11];

            if (c < 10) {
                sprintf(cent, "/cent_000");
            }
            else if ((100 > c) && (c >= 10)) {
                sprintf(cent, "/cent_00");
            }
            else {
                sprintf(cent, "/cent_0");
            }

            sprintf(numbuf, "%d", c);
            char outfile[128];
            sprintf(outfile, "%s%s%s%s", savedir, cent, numbuf, pgm);
            FILE *fout = fopen(outfile, "w+");
            if (fout == NULL) {
                printf("error creating file: %s\n", outfile);
                return;
            }

            fprintf(fout, "P2\n%d %d\n%3.0f\n", dim_root, dim_root, maxfeats);
            for (int d=0; d<dim; d++) {
                if ((d > 0) && (d % dim_root == 0)) {
                    fprintf(fout, "\n");
                }
                fprintf(fout, "%3.0f ", clust->features[c][d]);
            }

            fprintf(fout, "\n");
            fclose(fout);
        }
    }
    printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);
}

int main(int argc, char *argv[]) {
    // clock_t before = clock();

    if (argc < 3) {
        printf("usage: kmeans.exe <datafile> <nclust> [savedir] [maxiter]\n");
        return 0;
    }

    char* datafile = argv[1];
    int nclust = atoi(argv[2]);
    char* savedir = ".";
    int MAXITER = 100;

    if (argc > 3) {
        savedir = argv[3];
        mkdir(savedir, 0777);
        // mkdir(savedir, 0700);
    }

    if (argc > 4) {
        MAXITER = atoi(argv[4]);
    }

    printf("datafile: %s\nnclust: %d\nsavedir: %s\n", datafile, nclust, savedir);
    struct KMData *data = malloc(sizeof(struct KMData));
    data = kmdata_load(data, datafile);
    struct KMClust *clust = malloc(sizeof(struct KMClust));
    kmclust_new(clust, nclust, data->dim);

    printf("ndata: %d\ndim: %d\n\n", data->ndata, data->dim);

    // int c;
    #pragma omp parallel for
    for (int i=0; i<data->ndata; i++) {
        data->assigns[i] = i % clust->nclust;
        // c = i % clust->nclust;
        // data->assigns[i] = c;
    }
    
    // double icount;
    // int extra;
    #pragma omp parallel for
    for (int i=0; i<clust->nclust; i++) {
        double icount = data->ndata / clust->nclust;
        int extra = 0;
        if (i < data->ndata % clust->nclust) {
            extra = 1;
        }
        clust->counts[i] = icount + extra;
    }



    int curiter = 1;
    int nchanges = data->ndata;
    int best_clust;
    float best_distsq;
    float distsq = 0.0;

    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    int clust_counts[clust->nclust];
    // memset(clust_counts, 0, clust->nclust * sizeof(int));
    
    while (nchanges > 0 && curiter <= MAXITER) {

        #pragma omp parallel for
        for (int c=0; c<clust->nclust; c++) {   //Reset cluster centers to 0
            memset(clust->features[c], 0, sizeof(double) * clust->dim);
            // for (int d=0;d<clust->dim;d++) {
            //     clust->features[c][d] = 0.0;
            // }
        }
        // memset(clust->features, 0, sizeof(double) * clust->nclust * clust->dim);

        // int c;
        // double (*temp_features)[clust->nclust];
        // temp_features = malloc(sizeof(double[clust->nclust][clust->dim]));
        // memset(temp_features, 0, sizeof(double) * clust->nclust * clust->dim);

        // #pragma omp parallel for reduction(+:temp_features[:clust->nclust][:clust->dim])
        // int c;
        #pragma omp parallel for
        for (int i=0; i<data->ndata; i++) {     // sum data in clusters
            int c = data->assigns[i];
            for (int d=0; d<clust->dim; d++) {
                #pragma omp atomic
                clust->features[c][d] += data->features[i][d];
            }
        }

        #pragma omp parallel for
        for (int c=0; c<clust->nclust; c++) {   // divide ndatas of data to get mean
            if (clust->counts[c] > 0) {
                for (int d=0; d<clust->dim; d++) {
                    clust->features[c][d] = clust->features[c][d] / clust->counts[c];
                }
            }
        }
        // free(temp_features);

        // memset(clust->features, 0, sizeof(clust->features[0][0]) * clust->nclust * clust->dim);

        // double (*temp_features)[clust->nclust];
        // temp_features = malloc(sizeof(double[clust->nclust][clust->dim]));

        // int c;
        // double** temp_features;
        // temp_features = malloc(clust->nclust * sizeof(double*));
        // for (int i=0; i<clust->nclust; i++) {
        //     temp_features[i] = malloc(clust->dim * sizeof(double));
        // }

        // //allocate additional memory for each thread to have a "copy" of clust->features
        // //then do a reduction
        // // #pragma omp parallel for reduction(+:temp_features[:clust->nclust][:clust->dim])
        // #pragma omp parallel for
        // for (int i=0; i<data->ndata; i++) {     // sum data in clusters
        //     int c = data->assigns[i];
        //     for (int d=0; d<clust->dim; d++) {
        //         temp_features[c][d] += data->features[i][d];
        //         // clust->features[c][d] += data->features[i][d];
        //     }
        // }

        // #pragma omp parallel for
        // for (int c=0; c<clust->nclust; c++) {   // divide ndatas of data to get mean
        //     if (clust->counts[c] > 0) {
        //         for (int d=0; d<clust->dim; d++) {
        //             temp_features[c][d] /= clust->counts[c];
        //             // clust->features[c][d] = clust->features[c][d] / clust->counts[c];
        //         }
        //     }
        // }

        // #pragma omp parallel for
        // for (int i=0; i<clust->nclust; i++) {
        //     for (int j=0; j<clust->dim; j++) {
        //         clust->features[i][j] = temp_features[i][j];
        //     }
        // }

        // free(temp_features);

        // int c;
        // double (*temp_features)[clust->nclust];
        // temp_features = malloc(sizeof(double[clust->nclust][clust->dim]));

        // #pragma omp parallel reduction(+: temp_features[clust->nclust][clust->dim])
        // {
        //     #pragma omp for
        //     for (int i=0; i<data->ndata; i++) {     // sum data in clusters
        //         int c = data->assigns[i];
        //         for (int d=0; d<clust->dim; d++) {
        //             temp_features[c][d] += data->features[i][d];
        //         }
        //     }

        //     #pragma omp for
        //     for (int c=0; c<clust->nclust; c++) {   // divide ndatas of data to get mean
        //         if (clust->counts[c] > 0) {
        //             for (int d=0; d<clust->dim; d++) {
        //                 temp_features[c][d] /= clust->counts[c];
        //             }
        //         }
        //     }

        //     // #pragma omp for
        //     // for (int a=0; a<clust->nclust; a++) {
        //     //     for (int b=0; b<clust->dim; b++) {
        //     //         #pragma omp critical
        //     //         {
        //     //             clust->features[a][b] = temp_features[a][b];
        //     //         }
        //     //     }
        //     // }

        //     // free(temp_features);
        // }

        // #pragma omp parallel for
        // for (int i=0; i<clust->nclust; i++) {
        //     for (int j=0; j<clust->nclust; j++) {
        //         clust->features[i][j] = temp_features[i][j];
        //     }
        // }

        // free(temp_features);

        // for (int c=0; c<clust->nclust; c++) {   // reset cluster counts to 0
        //     clust->counts[c] = 0;
        // }
        memset(clust_counts, 0, clust->nclust * sizeof(int));
        memset(clust->counts, 0, sizeof(int) * clust->nclust);

        nchanges = 0;
        // candidate for parallelizing (data->ndata loop)
        #pragma omp parallel for reduction(+:nchanges) reduction(+:clust_counts[:clust->nclust])
        for (int i=0; i<data->ndata; i++) {
            best_clust = -1;
            best_distsq = INFINITY;
            for (int c=0; c<clust->nclust; c++) {
                distsq = 0.0;
                for (int d=0; d<clust->dim; d++) {
                    float diff = data->features[i][d] - clust->features[c][d];
                    distsq += diff * diff;
                }

                if (distsq < best_distsq) {
                    best_clust = c;
                    best_distsq = distsq;
                }
            }
            // need coordination for counts (use local variables)
            // clust->counts[best_clust] += 1;
            clust_counts[best_clust] += 1;
            if (best_clust != data->assigns[i]) {
                nchanges += 1;
                data->assigns[i] = best_clust;
            }   
        }

        #pragma omp parallel for
        for (int i=0; i<clust->nclust; i++) {
            clust->counts[i] += clust_counts[i];
        }

        printf(" %d:%6d |", curiter, nchanges);
        for (int i=0; i<clust->nclust; i++) {
            printf("%5d", clust->counts[i]);
        }
        printf("\n");
        curiter++;
    }

    printf("CONVERGED: after %d iterations\n", curiter);

    int confusion[data->nlabels][nclust];

    #pragma omp parallel for
    for (int i=0; i<data->nlabels; i++){
        for (int j=0; j<nclust; j++) {
            confusion[i][j] = 0;
        }
    }

    for (int i=0; i<data->ndata; i++) {
        confusion[data->labels[i]][data->assigns[i]] += 1;
    }

    printf("\n==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST\n");
    printf("   ");
    // printf("%5s"," ");
    for (int j=0; j<clust->nclust; j++) {
        printf("%5d", j);
    }
    printf("   TOT\n");

    int tot;
    for (int i=0; i<data->nlabels; i++) {
        printf(" %d:", i);
        tot = 0;
        for (int j=0; j<clust->nclust; j++) {
            printf("%5d", confusion[i][j]);
            tot += confusion[i][j];
        }
        printf("%5d\n", tot);
    }

    printf("TOT");
    tot = 0;

    for (int c=0; c<clust->nclust; c++) {
        printf("%5d", clust->counts[c]);
        tot += clust->counts[c];
    }

    printf(" %d \n", tot);

    char* labels = "/labels.txt";
    char outfile[128];
    sprintf(outfile, "%s%s", savedir, labels);
    FILE *fout = fopen(outfile, "w+");
    if (fout == NULL) {
        printf("error creating file: %s\n", outfile);
        return 1;
    }
    for (int i=0; i<data->ndata; i++) {
        fprintf(fout, "%2d %2d\n",data->labels[i], data->assigns[i]);
    }
    fclose(fout);
    printf("Saving cluster labels to file %s/labels.txt\n", savedir);

    save_pgm_files(clust, savedir);

    // Freeing stuff
    for (int i=0; i<data->ndata;i++) {
        free(data->features[i]);
    }
    free(data->features);
    free(data->assigns);
    free(data->labels);

    for (int i=0; i<clust->nclust; i++) {
        free(clust->features[i]);
    }

    free(clust->features);
    free(clust->counts);
    free(clust);
    free(data);

    // clock_t after = clock();
    // printf("Time= %ld", after - before);

    return 0;
}