#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>

struct KMData {
    int ndata;
    int dim;
    int* features;
    int* assigns;
    int* labels;
    int nlabels;
};

struct KMClust {
    int nclust;
    int dim;
    double* features;
    int* counts;
};

// More in depth CHECK() Macro, gives Cuda error code
// Found this via the internet, I did not write it
#define CHECK(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

void get_mem_info() {
    float free_m,total_m,used_m;
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);
    free_m =(uint)free_t/1048576.0 ;
    total_m=(uint)total_t/1048576.0;
    used_m=total_m-free_m;
    printf ( "  mem free %d .... %f MB mem total %d....%f MB mem used %f MB\n",free_t,free_m,total_t,total_m,used_m);
}

void transfer() {
    float* dev_tmp;
    CHECK(cudaMalloc((void**) &dev_tmp, sizeof(float)));
    float* host_tmp = (float*)malloc(sizeof(float));
    host_tmp[0] = 1.;
    CHECK(cudaMemcpy(dev_tmp, host_tmp, sizeof(float), cudaMemcpyHostToDevice));
    host_tmp[0] = 9.;
    CHECK(cudaMemcpy(host_tmp, dev_tmp, sizeof(float), cudaMemcpyDeviceToHost));
    printf("result of memcpy: %f\n", host_tmp[0]);

    cudaFree(dev_tmp);
    free(host_tmp);
}

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
    int line_size = (tot_tokens / tot_lines) - 2;
    int max_label = 0;
    data->labels = (int*)malloc(tot_lines * sizeof(int));
    data->features = (int*)malloc(tot_lines * line_size * sizeof(int));
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
            data->features[row * line_size + c - 1] = atoi(token);
            token = strtok(NULL, " ");
            c++;
        }
        row++;
    }
    data->assigns = (int*)malloc(tot_lines * sizeof(int));
    data->dim = line_size;
    data->nlabels = max_label + 1;
    fclose(fin);
    return data;
}

struct KMClust * kmclust_new(struct KMClust *clust, int nclust, int dim) {
    clust->nclust = nclust;
    clust->dim = dim;
    clust->features = (double*)malloc(nclust * dim * sizeof(double*));
    clust->counts = (int*)malloc(nclust * sizeof(int));
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
                maxfeats = (clust->features[j * clust->dim + i] > maxfeats) ? clust->features[j * clust->dim + i] : maxfeats;
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
                fprintf(fout, "%3.0f ", clust->features[c * clust->dim + d]);
            }

            fprintf(fout, "\n");
            fclose(fout);
        }
    }
    printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);
}

void create_cuda_clust_data(KMClust* clust, KMData* data, KMClust* dev_clust, KMData* dev_data) {    


    dev_clust->nclust = clust->nclust;
    dev_clust->dim = clust->dim;
    CHECK(cudaMalloc((void**) &dev_clust->features, clust->nclust * clust->dim * sizeof(double)));
    CHECK(cudaMemcpy(dev_clust->features, clust->features, clust->dim * clust->nclust * sizeof(double), cudaMemcpyHostToDevice));
    
    CHECK(cudaMalloc((void**) &dev_clust->counts, clust->nclust * sizeof(int)));
    CHECK(cudaMemcpy(dev_clust->counts, clust->counts, clust->nclust * sizeof(int), cudaMemcpyHostToDevice));
    

    dev_data->ndata = data->ndata;

    CHECK(cudaMalloc((void**) &dev_data->labels, data->ndata * sizeof(int)));
    CHECK(cudaMemcpy(dev_data->labels, data->labels, data->ndata * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void**) &dev_data->features, data->ndata * data->dim * sizeof(int)));
    CHECK(cudaMemcpy(dev_data->features, data->features, data->dim * data->ndata * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void**) &dev_data->assigns, data->ndata * sizeof(int)));
    CHECK(cudaMemcpy(dev_data->assigns, data->assigns, data->ndata * sizeof(int), cudaMemcpyHostToDevice));
    dev_data->dim = data->dim;
    dev_data->nlabels = data->nlabels;
}

void sync_device_host(KMClust* clust, KMClust* ptr_clust, KMData* data, KMData* ptr_data) {    
    CHECK(cudaMemcpy(clust->counts, ptr_clust->counts, clust->nclust * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(data->assigns, ptr_data->assigns, data->ndata * sizeof(int), cudaMemcpyDeviceToHost));
}

__global__ void new_cluster_centers(KMClust* clust, KMData* data) {
    long idx = threadIdx.x;
    if (idx < 784) {
        int c;

        for (int i=0; i<data->ndata; i++) {
            c = data->assigns[i];
            clust->features[c * clust->dim + idx] += data->features[i * clust->dim + idx];
        }

        for (int i=0; i<clust->nclust; i++) {
            if (clust->counts[i] > 0) {
                clust->features[i * clust->dim + idx] = clust->features[i * clust->dim + idx] / clust->counts[i];
            }
        }
    }
    __syncthreads();

}

__global__ void new_assignments(KMClust* clust, KMData* data, int numThreads, int* nchanges) {
    long idx = threadIdx.x + blockIdx.x * numThreads;
    if (idx < clust->nclust) {
        clust->counts[idx] = 0;
    }
    if (idx < data->ndata) {
        int best_clust = -1;
        float best_distsq = INFINITY;
        float distsq;
        for (int c=0; c<clust->nclust; c++) {
            distsq = 0.0;
            for (int d=0; d<clust->dim; d++) {
                float diff = data->features[idx * clust->dim + d] - clust->features[c * clust->dim + d];
                distsq += diff * diff;
            }

            if (distsq < best_distsq) {
                best_clust = c;
                best_distsq = distsq;
            }
        }
        atomicAdd(&clust->counts[best_clust], 1);
        if (best_clust != data->assigns[idx]) {
            atomicAdd(&nchanges[0], 1);
            data->assigns[idx] = best_clust;
        }
    }


    __syncthreads();
}

int main(int argc, char *argv[]) {
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
    struct KMData *data = (struct KMData*)malloc(sizeof(struct KMData));
    data = kmdata_load(data, datafile);
    struct KMClust *clust = (struct KMClust*)malloc(sizeof(struct KMClust));
    kmclust_new(clust, nclust, data->dim);

    printf("ndata: %d\ndim: %d\n\n", data->ndata, data->dim);

    int c;
    for (int i=0; i<data->ndata; i++) {
        c = i % clust->nclust;
        data->assigns[i] = c;
    }
    
    double icount;
    int extra;
    for (int i=0; i<clust->nclust; i++) {
        icount = data->ndata / clust->nclust;
        extra = 0;
        if (i < data->ndata % clust->nclust) {
            extra = 1;
        }
        clust->counts[i] = icount + extra;
    }

    struct KMClust* ptr_clust = (struct KMClust*)malloc(sizeof(KMClust));
    struct KMData* ptr_data = (struct KMData*)malloc(sizeof(KMData));
    create_cuda_clust_data(clust, data, ptr_clust, ptr_data);

    struct KMClust* dev_clust;
    struct KMData* dev_data;

    cudaMalloc((void**) &dev_clust, sizeof(KMClust));
    cudaMalloc((void**) &dev_data, sizeof(KMData));

    cudaMemcpy(dev_clust, ptr_clust, sizeof(KMClust), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data, ptr_data, sizeof(KMData), cudaMemcpyHostToDevice);




    int curiter = 1;
    int nchanges = data->ndata;

    int numThreads = 1024;
    int numBlocks = (data->ndata / numThreads) + 1;

    int* dev_nchanges;
    cudaMalloc((void**) &dev_nchanges, sizeof(int));

    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    int* iter;

    while (nchanges > 0 && curiter <= MAXITER) {

        // Set cluster features to 0
        CHECK(cudaMemset(ptr_clust->features, 0., clust->dim * clust->nclust * sizeof(double)));

        // Calculate new cluster centers
        new_cluster_centers<<<1, 784>>>(dev_clust, dev_data);

        // Ensure all threads finish and sync the GPU assigns and counts
        CHECK(cudaDeviceSynchronize());
        sync_device_host(clust, ptr_clust, data, ptr_data);
        
        nchanges = 0;
        CHECK(cudaMemset(dev_nchanges, 0, sizeof(int)));

        // Calculate new assignments
        new_assignments<<<numBlocks, numThreads>>>(dev_clust, dev_data, numThreads, dev_nchanges);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&nchanges, dev_nchanges, sizeof(int), cudaMemcpyDeviceToHost));

        sync_device_host(clust, ptr_clust, data, ptr_data);

        printf(" %d:%6d |", curiter, nchanges);
        for (int i=0; i<clust->nclust; i++) {
            printf("%5d", clust->counts[i]);
        }
        printf("\n");
        curiter++;
    }

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
    CHECK(cudaMemcpy(clust->features, ptr_clust->features, clust->nclust * clust->dim * sizeof(double), cudaMemcpyDeviceToHost));
    save_pgm_files(clust, savedir);
    // Freeing stuff

    free(data->features);
    free(data->assigns);
    free(data->labels);
    cudaFree(ptr_data->features);

    cudaFree(ptr_data->assigns);
    cudaFree(ptr_data->labels);

    free(clust->features);
    free(clust->counts);
    free(clust);
    free(data);

    cudaFree(ptr_clust->features);
    cudaFree(ptr_clust->counts);

    free(ptr_clust);
    free(ptr_data);
    cudaFree(dev_clust);
    cudaFree(dev_data);
    cudaFree(dev_nchanges);
    return 0;
}