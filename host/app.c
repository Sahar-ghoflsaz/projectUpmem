/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 * 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "../support/common.h"
#include "../support/matrix.h"
#include "../support/params.h"
#include "../support/partition.h"
#include "../support/timer.h"
#include "../support/utils.h"
#include "../support/aes.h"
//#include "../support/omp.h"
// Define the DPU Binary path as DPU_BINARY here.
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/spmv_dpu"
#endif

#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MB

/*
 * Main Structures:
 * 1. Matrices
 * 2. Input vector
 * 3. Output vector
 * 4. Help structures for data partitioning
 */
static struct RBDCSRMatrix* A;
static struct COOMatrix* B;
static val_dt* x;
static val_dt* z;
static val_dt* y;
static struct partition_info_t *part_info;


/**
 * @brief Specific information for each DPU
 */
struct dpu_info_t {
    uint32_t rows_per_dpu;
    uint32_t cols_per_dpu;
    uint32_t rows_per_dpu_pad;
    uint32_t prev_rows_dpu;
    uint32_t prev_nnz_dpu;
    uint32_t nnz;
    uint32_t nnz_pad;
    uint32_t ptr_offset; 
};

struct dpu_info_t *dpu_info;


/**
 * @brief find the dpus_per_row_partition
 * @param factor n to create partitions
 * @param column_partitions to create vert_partitions 
 * @param horz_partitions to return the 2D partitioning
 */
void find_partitions(uint32_t n, uint32_t *horz_partitions, uint32_t vert_partitions) {
    uint32_t dpus_per_vert_partition = n / vert_partitions;
    *horz_partitions = dpus_per_vert_partition;
}

/**
 * @brief initialize input vector 
 * @param pointer to input vector and vector size
 */
void init_vector(val_dt* vec, uint32_t size) {
    for(unsigned int i = 0; i < size; ++i) {
        vec[i] = (i%4+1);
    }
}

/**
 * @brief compute output in the host CPU
 */ 
static void spmv_host(val_dt* y, struct RBDCSRMatrix *A, val_dt* x) {
    uint64_t total_nnzs = 0;
    for (uint32_t c = 0; c < A->vert_partitions; c++) {
        for(uint32_t rowIndx = 0; rowIndx < A->nrows; ++rowIndx) {
            val_dt sum = 0;
            uint32_t ptr_offset = c * (A->nrows + 1);
            uint32_t col_offset = c * A->tile_width;
            for(uint32_t n = A->drowptr[ptr_offset + rowIndx]; n < A->drowptr[ptr_offset + rowIndx + 1]; n++) {
                uint32_t colIndx = A->dcolind[total_nnzs];    
                val_dt value = A->dval[total_nnzs++];    
                sum += x[col_offset + colIndx] * value;
            }
            y[rowIndx] += sum;
        }
    }
}


/**
 * @brief main of the host application
 */
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus; 
    uint32_t nr_of_ranks; 

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    DPU_ASSERT(dpu_get_nr_ranks(dpu_set, &nr_of_ranks));
    printf("[INFO] Allocated %d DPU(s)\n", nr_of_dpus);
    printf("[INFO] Allocated %d Rank(s)\n", nr_of_ranks);
    printf("[INFO] Allocated %d TASKLET(s) per DPU\n", NR_TASKLETS);

    unsigned int i;

    // Initialize input data 
    B = readCOOMatrix(p.fileName);
    printf("hi\n");
    sortCOOMatrix(B);
    uint32_t horz_partitions = 0;
    uint32_t vert_partitions = p.vert_partitions; 
    find_partitions(nr_of_dpus, &horz_partitions, p.vert_partitions);
    printf("[INFO] %dx%d Matrix Partitioning\n\n", horz_partitions, vert_partitions);
    A = coo2rbdcsr(B, horz_partitions, vert_partitions);
    freeCOOMatrix(B);

    // Initialize partition data
    part_info = partition_init(A, nr_of_dpus, p.max_nranks, NR_TASKLETS);
printf("hi\n");
#if FG_TRANS
    struct dpu_set_t rank;
    uint32_t each_rank;
    DPU_RANK_FOREACH(dpu_set, rank, each_rank){
        uint32_t nr_dpus_in_rank;
        DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_in_rank));
        part_info->active_dpus_per_rank[each_rank+1] = nr_dpus_in_rank;
    }
	
    uint32_t sum = 0;
    for(uint32_t i=0; i < p.max_nranks+1; i++) {
        part_info->accum_dpus_ranks[i] = part_info->active_dpus_per_rank[i] + sum;
        sum += part_info->active_dpus_per_rank[i];
    }

#endif
printf("hi\n");
    // Initialize help data - Padding needed
    uint32_t ncols_pad = A->ncols;
    uint32_t tile_width_pad = A->tile_width;
    uint32_t nrows_pad = A->nrows;
    if (ncols_pad % (8 / byte_dt) != 0)
        ncols_pad = ncols_pad + ((8 / byte_dt) - (ncols_pad % (8 / byte_dt)));
    if (tile_width_pad % (8 / byte_dt) != 0)
        tile_width_pad = tile_width_pad + ((8 / byte_dt) - (tile_width_pad % (8 / byte_dt)));
    if (nrows_pad % (8 / byte_dt) != 0)
        nrows_pad = nrows_pad + ((8 / byte_dt) - (nrows_pad % (8 / byte_dt)));

    // Allocate input vector
    x = (val_dt *) malloc(ncols_pad * sizeof(val_dt)); 
printf("in vec alloc\n");
    // Allocate output vector
    z = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 
printf("out vec alloc\n");
    // Initialize input vector with arbitrary data
    init_vector(x, ncols_pad);

    // Load-balance nnzs among DPUs of the same vertical partition
    partition_by_nnz(A, part_info);

    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t)); 
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    // Max limits for parallel transfers
    uint64_t max_rows_per_dpu = 0;
    uint64_t max_nnz_ind_per_dpu = 0;
    uint64_t max_nnz_val_per_dpu = 0;
    uint64_t max_rows_per_tasklet = 0;

    // Timer for measurements
    Timer timer;

    uint64_t total_nnzs = 0;
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // Find padding for rows and non-zero elements needed for CPU-DPU transfers
        uint32_t tile_horz_indx = i % A->horz_partitions; 
        uint32_t tile_vert_indx = i / A->horz_partitions; 
        uint32_t rows_per_dpu = part_info->row_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx + 1] - part_info->row_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx];
        uint32_t prev_rows_dpu = part_info->row_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx];

        // Pad data to be transfered
        uint32_t rows_per_dpu_pad = rows_per_dpu + 1;
        if (rows_per_dpu_pad % (8 / byte_dt) != 0)
            rows_per_dpu_pad += ((8 / byte_dt) - (rows_per_dpu_pad % (8 / byte_dt)));
#if INT64 || FP64
        if (rows_per_dpu_pad % 2 == 1)
            rows_per_dpu_pad++;
#endif
        if (rows_per_dpu_pad > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu_pad;

        unsigned int nnz, nnz_ind_pad, nnz_val_pad;
        nnz = A->drowptr[tile_vert_indx * (A->nrows + 1) + prev_rows_dpu + rows_per_dpu] - A->drowptr[tile_vert_indx * (A->nrows + 1) + prev_rows_dpu];
        if (nnz % 2 != 0)
            nnz_ind_pad = nnz + 1;
        else
            nnz_ind_pad = nnz;
        if (nnz % (8 / byte_dt) != 0)
            nnz_val_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
        else
            nnz_val_pad = nnz;

#if INT64 || FP64
        if (nnz_ind_pad % 2 == 1)
            nnz_ind_pad++;
        if (nnz_val_pad % 2 == 1)
            nnz_val_pad++;
#endif
        if (nnz_ind_pad > max_nnz_ind_per_dpu)
            max_nnz_ind_per_dpu = nnz_ind_pad;
        if (nnz_val_pad > max_nnz_val_per_dpu)
            max_nnz_val_per_dpu = nnz_val_pad;

        uint32_t prev_nnz_dpu = total_nnzs;
        total_nnzs += nnz;

        // Keep information per DPU
        dpu_info[i].rows_per_dpu = rows_per_dpu;
        dpu_info[i].cols_per_dpu = A->tile_width;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;
        dpu_info[i].prev_nnz_dpu = prev_nnz_dpu;
        dpu_info[i].nnz = nnz;
        dpu_info[i].nnz_pad = nnz_ind_pad;
        dpu_info[i].ptr_offset = tile_vert_indx * (A->nrows + 1) + prev_rows_dpu;

        // Find input arguments per DPU
        input_args[i].nrows = rows_per_dpu;
        input_args[i].tcols = tile_width_pad; 
        input_args[i].nnz_pad = nnz_ind_pad;
        input_args[i].nnz_offset = A->drowptr[tile_vert_indx * (A->nrows + 1) + prev_rows_dpu];

#if BLNC_TSKLT_ROW
        // Load-balance rows across tasklets 
        partition_tsklt_by_row(part_info, i, rows_per_dpu, NR_TASKLETS);
#else
        // Load-balance nnz across tasklets 
        partition_tsklt_by_nnz(A, part_info, i, rows_per_dpu, nnz, tile_vert_indx * (A->nrows + 1) + prev_rows_dpu, NR_TASKLETS);
#endif

        uint32_t t;
        for (t = 0; t < NR_TASKLETS; t++) {
            // Find input arguments per tasklet
            input_args[i].start_row[t] = part_info->row_split_tasklet[t]; 
            input_args[i].rows_per_tasklet[t] = part_info->row_split_tasklet[t+1] - part_info->row_split_tasklet[t];
            if (input_args[i].rows_per_tasklet[t] > max_rows_per_tasklet)
                max_rows_per_tasklet = input_args[i].rows_per_tasklet[t];
        }


    }
    assert(A->nnz == total_nnzs && "wrong balancing");

#if FG_TRANS
    // Find max number of rows (subset of elements of the output vector) among DPUs of each rank
    DPU_RANK_FOREACH(dpu_set, rank, each_rank){
        uint32_t max_rows_cur_rank = 0;
        uint32_t nr_dpus_in_rank;
        DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_in_rank));
        uint32_t start_dpu = part_info->accum_dpus_ranks[each_rank];
        for (uint32_t k = 0; k < nr_dpus_in_rank; k++) {
            if (start_dpu + k >= nr_of_dpus)
                break;
            if (dpu_info[start_dpu + k].rows_per_dpu > max_rows_cur_rank)
                max_rows_cur_rank =  dpu_info[start_dpu + k].rows_per_dpu;

        }
        if (max_rows_cur_rank % 2 != 0)
            max_rows_cur_rank++;
        if (max_rows_cur_rank % (8 / byte_dt)  != 0) 
            max_rows_cur_rank += ((8 / byte_dt) - (max_rows_cur_rank % (8 / byte_dt)));
        part_info->max_rows_per_rank[each_rank] = (uint32_t) max_rows_cur_rank;
    }
#endif


    // Initializations for parallel transfers with padding needed
    if (max_rows_per_dpu % 2 != 0)
        max_rows_per_dpu++;
    if (max_rows_per_dpu % (8 / byte_dt) != 0)
        max_rows_per_dpu += ((8 / byte_dt) - (max_rows_per_dpu % (8 / byte_dt)));
    if (max_nnz_ind_per_dpu % 2 != 0)
        max_nnz_ind_per_dpu++;
    if (max_nnz_val_per_dpu % (8 / byte_dt) != 0)
        max_nnz_val_per_dpu += ((8 / byte_dt) - (max_nnz_val_per_dpu % (8 / byte_dt)));
    if (max_rows_per_tasklet % (8 / byte_dt) != 0)
        max_rows_per_tasklet += ((8 / byte_dt) - (max_rows_per_tasklet % (8 / byte_dt)));

    // Re-allocations for padding needed
    A->drowptr = (uint32_t *) realloc(A->drowptr, (max_rows_per_dpu * (uint64_t) nr_of_dpus * sizeof(uint32_t)));
    A->dcolind = (uint32_t *) realloc(A->dcolind, (max_nnz_ind_per_dpu * nr_of_dpus * sizeof(uint32_t)));
    A->dval = (val_dt *) realloc(A->dval, (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt)));
    x = (val_dt *) realloc(x, (uint64_t) ((uint64_t) A->vert_partitions * (uint64_t) tile_width_pad) * (uint64_t) sizeof(val_dt)); 
    y = (val_dt *) malloc((uint64_t) ((uint64_t) nr_of_dpus * (uint64_t) max_rows_per_dpu) * (uint64_t) sizeof(val_dt)); 

    // Count total number of bytes to be transfered in MRAM of DPU
    unsigned long int total_bytes;
    total_bytes = ((max_rows_per_dpu) * sizeof(uint32_t)) + (max_nnz_ind_per_dpu * sizeof(uint32_t)) + (max_nnz_val_per_dpu * sizeof(val_dt)) + (tile_width_pad * sizeof(val_dt)) + (max_rows_per_dpu * sizeof(val_dt));
    assert(total_bytes <= DPU_CAPACITY && "Bytes needed exceeded MRAM size");
    Timer timertot;
    startTimer(&timertot,1);

    // Copy input arguments to DPUs
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        input_args[i].max_rows = max_rows_per_dpu; 
        input_args[i].max_nnz_ind = max_nnz_ind_per_dpu; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

printf("arg copy \n");
    // Copy input matrix to DPUs
    startTimer(&timer, 0);
printf("input copy \n");
    // Copy Rowptr 
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->drowptr + dpu_info[i].ptr_offset));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (max_rows_per_dpu * sizeof(val_dt) + tile_width_pad * sizeof(val_dt)), max_rows_per_dpu * sizeof(uint32_t), DPU_XFER_DEFAULT));
printf("rowptr copy \n");
    // Copy Colind
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->dcolind + dpu_info[i].prev_nnz_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt) + tile_width_pad * sizeof(val_dt) + max_rows_per_dpu * sizeof(uint32_t), max_nnz_ind_per_dpu * sizeof(uint32_t), DPU_XFER_DEFAULT));
printf("col copy \n");
		
		/*****neeeeeeeeeeeeeeeew added code********************/

	//(A->dval, (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt))	
    uint8_t *first = (uint8_t *) malloc((uint64_t) (max_nnz_val_per_dpu * nr_of_dpus * sizeof(uint8_t))); 
	val_dt *ciphertext = (val_dt *) malloc((uint64_t) (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt))); 
	val_dt *temp = (val_dt *) malloc((uint64_t) (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt))); 
    uint8_t *verificationTag = (val_dt *) malloc((uint64_t) (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt)/16)); 
    uint8_t *s = (val_dt *) malloc((uint64_t) (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt)/16)); 

	for(int i=0;i< max_nnz_val_per_dpu * nr_of_dpus; i++){
       // for( int i=0;i<16; i++){
		first[i] = A->dval + (sizeof(val_dt)*i);
        
		//printf("%d  \n", ciphertext[i]);
	//}
	//	ciphertext[i] = A->dval[i];
		//printf("%d \n ", ciphertext[i]);
	}
    //for(int i=0;i< (max_nnz_val_per_dpu * nr_of_dpus)/16; i++){
       // for( int i=0;i<16; i++){
	//	s[i] = A->dval + (sizeof(val_dt)*128*i);
        
		//printf("%d  \n", ciphertext[i]);
	//}
	//	ciphertext[i] = A->dval[i];
		//printf("%d \n ", ciphertext[i]);
	//}
	printf("test1\n");
	uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	uint8_t iv[]  = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
	/*struct AES_ctx ctx;
	printf("test2\n");
	AES_init_ctx_iv(&ctx, key, iv);
    	//uint32_t buffercopy[] = buffer[];
    	printf("test3\n");
    	AES_CBC_encrypt_buffer(&ctx, ciphertext, (max_nnz_val_per_dpu * nr_of_dpus));*/

    	printf("CBC encrypt: ");

	struct AES_ctx ctx;

    AES_init_ctx(&ctx, key);
    AES_ECB_encrypt(&ctx, first);

    //AES_init_ctx(&ctx, key);
    //AES_ECB_encrypt(&ctx, s);
	/*for( int i=0;i<NR_ELEM_PER_DPU; i++){
	
    
    	printf("%d  ", cipertext[i]);
	}*/
	
	/* if (0 == memcmp((char*) out, (char*) ciphertext, 128)) {
	 	printf("SUCCESS!\n");
   	 } else{
       	printf("FAILURE!\n");
    	 }*/
	
	//uint8_t resultFinal[NR_ELEM_PER_DPU]= {0};
	printf("done\n");
	for(int i=0;i< max_nnz_val_per_dpu * nr_of_dpus; i++){

        ciphertext[i] = (val_dt)first[i];
		temp[i]= A->dval[i]- ciphertext[i];	
        
	}
    //uint8_t *tempver = (val_dt *) malloc((uint64_t) (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt)/16)); 
    //for(int i=0;i< (max_nnz_val_per_dpu * nr_of_dpus)/16; i++){
      //  for( int j=0;j<16; j++){
		//    tempver[i] += pow(s,16-j) * ciphertext[i+j];
       // }
        //verificationTag[i] = tempver[i];
		
	//}

    //AES_init_ctx(&ctx, key);
    //AES_ECB_encrypt(&ctx, tempver);

    //for(int i=0;i< max_nnz_val_per_dpu * nr_of_dpus/16; i++){

      //  finalCrypTag[i] = (val_dt)tempver[i];
		//tempCal[i]= verificationTag[i]- finalCrypTag[i];	
        
//	}

	/************end of neeeeeeeeeeeeew added code****************/
	
printf("cipher done\n");
    // Copy Values
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, ciphertext + dpu_info[i].prev_nnz_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt) + tile_width_pad * sizeof(val_dt) + max_rows_per_dpu * sizeof(uint32_t) + max_nnz_ind_per_dpu * sizeof(uint32_t), max_nnz_val_per_dpu * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0);


printf("copy val done\n");
    // Copy input vector  to DPUs
    startTimer(&timer, 1);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t tile_vert_indx = i / A->horz_partitions; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, x + tile_vert_indx * A->tile_width));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt), tile_width_pad * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 1);


printf("copy vec done\n");

  //  DPU_FOREACH(set,dpu,each_dpu){
	//	DPU_ASSERT(dpu_copy_to(dpu,"finalCrypTag",0, &finalCrypTag[each_dpu  * NR_ELEM_PER_DPU],max_nnz_val_per_dpu * nr_of_dpus/16);
	//}


    // Run kernel on DPUs
    startTimer(&timer, 2);
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    stopTimer(&timer, 2);
	//printf("run kernel done\n");
	/*********neeeeeeeeeeeeeeeeeeeeeeeeeeew **************/
	
	startTimer(&timer, 4);
    val_dt *hostResult = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 
    //spmv_host(y_host, A, x); 
    printf("sahar\n");
    
    uint64_t total_nnzs1 = 0;
    for (uint32_t c = 0; c < A->vert_partitions; c++) {
        for(uint32_t rowIndx = 0; rowIndx < A->nrows; ++rowIndx) {
            val_dt sum = 0;
            uint32_t ptr_offset = c * (A->nrows + 1);
            uint32_t col_offset = c * A->tile_width;
            for(uint32_t n = A->drowptr[ptr_offset + rowIndx]; n < A->drowptr[ptr_offset + rowIndx + 1]; n++) {
                uint32_t colIndx = A->dcolind[total_nnzs1];    
                val_dt value = temp[total_nnzs1++];    
                sum += x[col_offset + colIndx] * value;
            }
            hostResult[rowIndx] += sum;
        }
    }
    
    stopTimer(&timer, 4);
    
    printf("temp cal done\n");
    /****************end of new *************************/	
	//DPU_FOREACH(set,dpu,each_dpu){
	//	DPU_ASSERT(dpu_copy_from(dpu, "tagResult", 0, &tagResult[each_dpu][0], sizeof(double)*NR_ELEM_PER_DPU));
	//	DPU_ASSERT(dpu_log_read(dpu, stdout));
	//}
#if LOG
    // Display DPU Log (default: disabled)
    DPU_FOREACH(dpu_set, dpu) {
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif
    //while(dpu_status==0);
    //stopTimer(&timer, 2);
    //printf("run kernel done\n");
    // Retrieve results for output vector from DPUs
    startTimer(&timer, 3);
#if CG_TRANS
    // Coarse-grained data transfers in the output vector
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * max_rows_per_dpu)));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * sizeof(val_dt), DPU_XFER_DEFAULT));
#endif

#if FG_TRANS
    // Fine-grained data transfers in the output vector at rank granularity
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * max_rows_per_dpu)));
    }
    i = 0;
    //struct dpu_set_t rank;
    DPU_RANK_FOREACH(dpu_set, rank) {
        DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, part_info->max_rows_per_rank[i] * sizeof(val_dt), DPU_XFER_ASYNC));
        i++;
    }
    DPU_ASSERT(dpu_sync(dpu_set));
#endif
    stopTimer(&timer, 3);
    //while(dpu_status==1);
    //stopTimer(&timer, 2);
    printf("run kernel done and retrived\n");


    // Merge partial results to the host CPU
    startTimer(&timer, 5);
    uint32_t r, c, t;
    for (c = 0; c < A->vert_partitions; c++) {
        for (r = 0; r < A->horz_partitions; r++) {
#pragma omp parallel for num_threads(p.nthreads) shared(A, z, y, max_rows_per_dpu, c, r) private(t)
            for (t = 0; t < part_info->row_split[c * (A->horz_partitions + 1) + r+1] - part_info->row_split[c * (A->horz_partitions + 1) + r]; t++) {
                z[part_info->row_split[c * (A->horz_partitions + 1) + r] + t] += y[(c * A->horz_partitions + r) * max_rows_per_dpu + t];
            }
        }
    }
    /*neeeeeeeeeeeeeeeeeeeeeeew*/
    printf("merge done\n");
    val_dt *total = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 
    
    for (i = 0; i < A->nrows; i++) {
        total[i] = hostResult[i] + z[i];
    }
    stopTimer(&timer, 5);
printf("sum done\n");
    stopTimer(&timertot,1);

	/************end of new ***************/

    
    // Print timing results
    printf("\n");
    printf("Load Matrix ");
    printTimer(&timer, 0);
    printf("Load Input Vector ");
    printTimer(&timer, 1);
    printf("Kernel ");
    printTimer(&timer, 2);
    printf("Retrieve Output Vector ");
    printTimer(&timer, 3);
    printf("SecCPU Computation ");
    printTimer(&timer, 4);
    //printf("\n\n");
    printf("Merge Partial Results ");
    printTimer(&timer, 5);
    printf("Total ");
    printTimer(&timertot, 1);
    printf("\n\n");

//#if CHECK_CORR
    // Check output
    //startTimer(&timer, 4);
    val_dt *y_host = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 
    startTimer(&timer,6);
    spmv_host(y_host, A, x); 
	stopTimer(&timer,6);

    /*for (i = 0; i < A->nrows; i++) {
        printf(" plaintext: %f   ", A->dval[i]);
        printf(" ciphertext: %f   ", ciphertext[i]);
        printf(" temp: %f   \n", temp[i]);
        printf(" host result: %f   ", hostResult[i]);
        printf(" DPU Result: %f   \n", z[i]);
         //printf(" expected Result: %d   \n", y_host[i]);
    }*/

	printf("Only CPU Calculation ");
    printTimer(&timer, 6);

    bool status = true;
    i = 0;
    for (i = 0; i < A->nrows; i++) {
        if(y_host[i] != total[i]) {
            status = false;
        }
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    free(y_host);
    free(temp);
    free(ciphertext);
    free(hostResult);
    free(total);
//#endif


    // Deallocation
    freeRBDCSRMatrix(A);
    free(x);
    free(z);
    free(y);
    partition_free(part_info);
    DPU_ASSERT(dpu_free(dpu_set));

    return 0;

}
