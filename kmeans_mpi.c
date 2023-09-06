#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <mpi.h>

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

int filestats(char *filename, long unsigned *tot_tokens, long unsigned *tot_lines){
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

  long unsigned ntokens=0, nlines=0, column=0;
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

    int total_procs, proc_id, host_len;
    char host[256];
    MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
    MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 
    MPI_Get_processor_name(host, &host_len);      // get the symbolic host name 

    // ssize_t tot_tokens = malloc(sizeof(ssize_t));
    // ssize_t tot_lines = malloc(sizeof(ssize_t));
    // unsigned long tot_tokens = malloc(sizeof(unsigned long));
    // unsigned long tot_lines = malloc(sizeof(unsigned long));
    // int max_label = malloc(sizeof(int));
    unsigned long tot_tokens;
    unsigned long tot_lines;
    int max_label;
    data->ndata = 0;

    if (proc_id == 0) {
        int stat = filestats(datafile, &tot_tokens, &tot_lines);
        if (stat == -1) {
            printf("filestats return stat: %d", stat);
            return data;
        }

        FILE *fin = fopen(datafile, "r");
        if (fin == NULL) { printf("error opening file\n"); }
        data->ndata = tot_lines;
        int line_size = tot_tokens / tot_lines;
        max_label = 0;
        MPI_Bcast(&tot_tokens, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); // tot_tokens
        MPI_Bcast(&tot_lines, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); // tot_lines
        data->labels = (int*) malloc((int)tot_lines * sizeof(int));
        data->features = malloc(tot_lines * sizeof(int*));
        for (int i=0; i<tot_lines; i++) {
            data->features[i] = malloc(line_size * sizeof(int));
        }
        char line[3142];
        int row = 0;
        int c;
        while (fgets(line, 3142*sizeof(char), fin) != NULL) {
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
        int tot_lines_int = (int) tot_lines;
        MPI_Bcast(data->labels, tot_lines_int, MPI_INT, 0, MPI_COMM_WORLD);    // for data->labels
        for (int i=0; i<tot_lines_int; i++) {
            MPI_Bcast(data->features[i], line_size, MPI_INT, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(&max_label, 1, MPI_INT, 0, MPI_COMM_WORLD);    // for max_label
        data->assigns = malloc(tot_lines * sizeof(int));
        data->dim = line_size - 2;
        data->nlabels = max_label + 1;
        fclose(fin);
    }
    else {
        max_label = 0;
        MPI_Bcast(&tot_tokens, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); // tot_lines
        MPI_Bcast(&tot_lines, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); // line_size
        int line_size = tot_tokens / tot_lines;
        data->labels = malloc(tot_lines * sizeof(int));
        data->features = malloc(tot_lines * sizeof(int*));
        for (int i=0; i<tot_lines; i++) {
            data->features[i] = malloc(line_size * sizeof(int));
        }
        int tot_lines_int = (int) tot_lines;
        MPI_Bcast(data->labels, tot_lines_int, MPI_INT, 0, MPI_COMM_WORLD);    // for data->labels
        for (int i=0; i<tot_lines_int; i++) {
            MPI_Bcast(data->features[i], line_size, MPI_INT, 0, MPI_COMM_WORLD);
        }
        // MPI_Bcast(&(data->features[0][0]), features_size, MPI_INT, 0, MPI_COMM_WORLD);    // for data->features
        MPI_Bcast(&max_label, 1, MPI_INT, 0, MPI_COMM_WORLD);
        fflush(stdout);
        data->assigns = malloc(tot_lines * sizeof(int));
        data->dim = line_size - 2;
        data->nlabels = max_label + 1;
        data->ndata = tot_lines;
    }


    // MPI_Bcast(data, num_elements, MPI_INT, root_proc,
    // MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    return data;
}

struct KMClust * kmclust_new(struct KMClust *clust, int nclust, int dim) {
    clust->nclust = nclust;
    clust->dim = dim;
    // clust->counts = malloc(nclust * sizeof(int));
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
        for (int i=0; i<dim; i++) {
            for (int j=0; j<nclust; j++) {
                maxfeats = (clust->features[j][i] > maxfeats) ? clust->features[j][i] : maxfeats;
                // maxfeats = (int) maxfeats;
            }
        }
        
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
                // fprintf(fout, "%4d", (int)(clust->features[c][d]));
            }
            fprintf(fout, "\n");
            fclose(fout);
        }
    }
    printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("usage: mpirun -np <num_procs> ./kmeans_mpi <datafile> <nclust> [savedir] [maxiter]\n");
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

    int total_procs, proc_id, host_len;
    char host[256];

    MPI_Init (&argc, &argv);                      // starts MPI 
    MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
    MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 
    MPI_Get_processor_name(host, &host_len);      // get the symbolic host name 

    // if (argc < 3) {
    //     printf("usage: kmeans.exe <datafile> <nclust> [savedir] [maxiter]\n");
    //     return 0;
    // }

    // char* datafile = argv[1];
    // int nclust = atoi(argv[2]);
    // char* savedir = ".";
    // int MAXITER = 100;

    // if (argc > 3) {
    //     savedir = argv[3];
    //     // mkdir(savedir, 0700);
    // }

    // if (argc > 4) {
    //     MAXITER = atoi(argv[4]);
    // }

    if (proc_id == 0) {
        printf("datafile: %s\nnclust: %d\nsavedir: %s\n", datafile, nclust, savedir);
    }
    struct KMData *data = malloc(sizeof(struct KMData));
    struct KMClust *clust = malloc(sizeof(struct KMClust));

    data = kmdata_load(data, datafile);
    kmclust_new(clust, nclust, data->dim);

    // MPI_Barrier(MPI_COMM_WORLD);

    /* CODE SNIPPET BORROWED FROM LECTURES */
    int total_ndata_elements = data->ndata;
    int total_nclust_elements = clust->nclust;
    int total_dim_elements = clust->dim;
    int (*counts)[total_procs];
    int (*displs)[total_procs];
    counts = malloc(sizeof(int[3][total_procs]));
    displs = malloc(sizeof(int[3][total_procs]));
    int ndata_elements_per_proc = total_ndata_elements / total_procs;
    int nclust_elements_per_proc = total_nclust_elements / total_procs;
    int dim_elements_per_proc = total_dim_elements / total_procs;
    int ndata_surplus = total_ndata_elements % total_procs;
    int nclust_surplus = total_nclust_elements % total_procs;
    int dim_surplus = total_dim_elements % total_procs;
    

    // counts[0] = ndata per proc
    // counts[1] = nclust per proc
    // counts[2] = dim per proc
    for(int i=0; i<total_procs; i++) {
        counts[0][i] = (i < ndata_surplus) ? ndata_elements_per_proc + 1 : ndata_elements_per_proc;
        // printf("counts[0][%d]= %d\n", i, counts[0][i]);
        counts[1][i] = (i < nclust_surplus) ? nclust_elements_per_proc + 1 : nclust_elements_per_proc;
        // printf("counts[1][%d]= %d\n", i, counts[1][i]);
        counts[2][i] = (i < dim_surplus) ? dim_elements_per_proc + 1 : dim_elements_per_proc;
        // printf("counts[2][%d]= %d\n", i, counts[2][i]);
        displs[0][i] = (i == 0) ? 0 : displs[0][i-1] + counts[0][i-1];
        // printf("displs[0][%d]= %d\n", i, displs[0][i]);
        displs[1][i] = (i == 0) ? 0 : displs[1][i-1] + counts[1][i-1];
        // printf("displs[1][%d]= %d\n", i, displs[1][i]);
        displs[2][i] = (i == 0) ? 0 : displs[2][i-1] + counts[2][i-1];
        // printf("displs[2][%d]= %d\n", i, displs[2][i]);
    }
    /* CODE SNIPPET BORROWED FROM LECTURES */


    if (proc_id == 0) {
        printf("ndata: %d\ndim: %d\n\n", data->ndata, data->dim);
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    int ndata = total_ndata_elements;
    nclust = total_nclust_elements;
    // int dim = total_dim_elements;
    // int *assigns = malloc(sizeof(int) * counts[0][proc_id]);
    for (int i=displs[0][proc_id]; i<counts[0][proc_id] + displs[0][proc_id]; i++) {
        data->assigns[i] = i % nclust;
    }

    // For data->assigns
    MPI_Allgatherv(MPI_IN_PLACE, counts[0][proc_id], MPI_INT, data->assigns,
        counts[0], displs[0], MPI_INT, MPI_COMM_WORLD);

    // int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //                void *recvbuf, const int *recvcounts, const int *displs,
    //                MPI_Datatype recvtype, MPI_Comm comm)

    double icount = ndata / nclust;
    int extra;
    int clust_counts[counts[1][proc_id]];
    for (int i=0; i<counts[1][proc_id]; i++) {
        extra = ((i + displs[1][proc_id]) < (ndata % nclust)) ? 1 : 0;
        clust_counts[i] = icount + extra;
    }

    // For clust->counts

    MPI_Allgatherv(clust_counts, counts[1][proc_id], MPI_INT, clust->counts,
        counts[1], displs[1], MPI_INT, MPI_COMM_WORLD);
    // printf("proc_id %d clust counts: %d\n", proc_id, clust_counts[0]);


    int curiter = 1;
    int nchanges = ndata;
    int best_clust;
    float best_distsq;
    float distsq = 0.0;

    if (proc_id == 0) {
        printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
        printf("ITER NCHANGE CLUST_COUNTS\n");
    }

    while (nchanges > 0 && curiter <= MAXITER) {

        for (int c=0; c<clust->nclust; c++) {   //Reset cluster centers to 0
            for (int d=0;d<clust->dim;d++) {
                clust->features[c][d] = 0.0;
            }
        }

        int c;
        for (int i=displs[0][proc_id]; i<counts[0][proc_id]+displs[0][proc_id]; i++) {     // sum data in clusters

            c = data->assigns[i];

            for (int d=0; d<clust->dim; d++) {
                clust->features[c][d] += data->features[i][d];


            }
        }

        for (int i=0; i<nclust; i++) {
            MPI_Allreduce(MPI_IN_PLACE, clust->features[i], clust->dim, MPI_DOUBLE, 
            MPI_SUM, MPI_COMM_WORLD);
        }
        for (int c=0; c<nclust; c++) {   // divide ndatas of data to get mean
            if (clust->counts[c] > 0) {
                for (int d=0; d<clust->dim; d++) {
                    clust->features[c][d] /= clust->counts[c];
                }
            }
        }        

        for (int c=0; c<clust->nclust; c++) {   // reset cluster counts to 0
            clust->counts[c] = 0;
        }

        nchanges = 0;

        for (int i=displs[0][proc_id]; i<counts[0][proc_id] + displs[0][proc_id]; i++) {
            best_distsq = INFINITY;
            best_clust = -1;
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
            clust->counts[best_clust] += 1;
            if (best_clust != data->assigns[i]) {
                nchanges += 1;
                data->assigns[i] = best_clust;
            }   
        }

        MPI_Allgatherv(MPI_IN_PLACE, counts[0][proc_id], MPI_INT, data->assigns,
                        counts[0], displs[0], MPI_INT, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, clust->counts, nclust, MPI_INT, 
                        MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &nchanges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (proc_id == 0) {
            printf(" %d:%6d |", curiter, nchanges);
            for (int i=0; i<clust->nclust; i++) {
                printf("%5d", clust->counts[i]);
            }
            printf("\n");
        }

        curiter++;

    }



    if (proc_id == 0) {
        printf("CONVERGED: after %d iterations\n", curiter);

        int confusion[data->nlabels][nclust];
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


    //   # LABEL FILE OUTPUT
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
    }

    // Freeing stuff
    for (int i=0; i<ndata;i++) {
        free(data->features[i]);
    }
    free(data->features);
    free(data->assigns);
    free(data->labels);

    for (int i=0; i<nclust; i++) {
        free(clust->features[i]);
    }

    free(clust->features);
    free(clust->counts);
    free(clust);
    free(data);
    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}