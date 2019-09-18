#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <mpi.h>
#include "libs/bitmap.h"

// Convolutional Kernel Examples, each with dimension 3, gaussian kernel with dimension 5.
// If you apply another kernel, remember not only to exchange the kernel but also the kernelFactor and the correct dimension.
int const sobelYKernel[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYKernelFactor = (float) 1.0;

int const sobelXKernel[] = {-1, -0, -1,
                            -2,  0, -2,
                            -1,  0, -1 , 0};
float const sobelXKernelFactor = (float) 1.0;

int const laplacian1Kernel[] = {-1,  -4,  -1,
                                -4,  20,  -4,
                                -1,  -4,  -1};
float const laplacian1KernelFactor = (float) 1.0;

int const laplacian2Kernel[] = {0,  1,  0,
                                1, -4,  1,
                                0,  1,  0};
float const laplacian2KernelFactor = (float) 1.0;

int const laplacian3Kernel[] = {-1,  -1,  -1,
                                -1,   8,  -1,
                                -1,  -1,  -1};
float const laplacian3KernelFactor = (float) 1.0;

// Bonus Kernel:
int const gaussianKernel[] = {1,  4,  6,  4, 1,
                              4, 16, 24, 16, 4,
                              6, 24, 36, 24, 6,
                              4, 16, 24, 16, 4,
                              1,  4,  6,  4, 1 };

float const gaussianKernelFactor = (float) 1.0 / 256.0;

// Helper function to swap bmpImageChannel pointers
void swapImageChannel(bmpImageChannel **one, bmpImageChannel **two) {
  bmpImageChannel *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(unsigned char **out, unsigned char **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor) {
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++) {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++) {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[yy][xx] * kernel[nky * kernelDim + nkx];
        }
      }
      aggregate *= kernelFactor;
      if (aggregate > 0) {
        out[y][x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y][x] = 0;
      }
    }
  }
}


void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL); // MPI Initialization
	
  // Get the id of each process and the total amount of processes
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	
  // Variables to hold the dimensions of the original image
  int image_height; 
  int image_width;
  int image_res;
	
  // Global image channel and local imageChannel for each process chunk
  bmpImageChannel *imageChannel = newBmpImageChannel(0, 0);
  bmpImageChannel *localImageChannel = newBmpImageChannel(0, 0);
  
  unsigned char *data_1D = NULL;
  unsigned char *localData_1D = NULL;
	 	
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  int ret = 0;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hi:";
  {
    char *endptr;
    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        goto graceful_exit;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    goto error_exit;
  }
  input = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */
	   
  // Root proceess reads the image
  if (my_rank == 0) {
    /*
	Create the BMP image and load it from disk.
	*/
	bmpImage *image = newBmpImage(0,0);
	if (image == NULL) {
	  fprintf(stderr, "Could not allocate new image!\n");
	  goto error_exit;
	}

	if (loadBmpImage(image, input) != 0) {
	  fprintf(stderr, "Could not load bmp image '%s'!\n", input);
	  freeBmpImage(image);
	  goto error_exit;
	}

    // Create a single color channel image. It is easier to work just with one color
    imageChannel = newBmpImageChannel(image->width, image->height);
    if (imageChannel == NULL) {
      fprintf(stderr, "Could not allocate new image channel!\n");
      freeBmpImage(image);
      goto error_exit;
    }

    // Extract from the loaded image an average over all colors - nothing else than
    // a black and white representation
    // extractImageChannel and mapImageChannel need the images to be in the exact
    // same dimensions!
    // Other prepared extraction functions are extractRed, extractGreen, extractBlue
    if(extractImageChannel(imageChannel, image, extractAverage) != 0) {
      fprintf(stderr, "Could not extract image channel!\n");
      freeBmpImage(image);
      freeBmpImageChannel(imageChannel);
      goto error_exit;
    }
  
    // Assign values to image dimensions so all processes can know the original image dimensions
    image_height = imageChannel->height;
    image_width  = imageChannel->width;
    
    data_1D = (unsigned char*) calloc(image_width * image_height, sizeof(unsigned char));
	for (unsigned int i = 0; i < image_height; i++) {
    for (unsigned int j = 0; j < image_width; j++) {
      data_1D[(i * image_width) + j] = imageChannel->data[i][j];
      //data_1D[(i * image_width) + j] = 0;
    }
  }
  }

  // Every process knows now the image dimensions
  MPI_Bcast(&image_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&image_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  image_res = image_height * image_height;
  //MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("Broadcast successful. Iterations = %d, Image Width = %d and Image Height = %d\n", iterations, image_width, image_height);
  
  // For each process, create a buffer that will hold a subset of the original image
  int chunk_height = (image_height / num_proc)  + 3; // 2 extra rows for shared borders and 1 extra row in case the data can't be equally divided
  int chunk_size = chunk_height * image_width;
  unsigned char *subset = (unsigned char*) calloc(chunk_size, sizeof(unsigned char));
  printf("Chunk allocated successfully. Height = %d and Size = %d\n", chunk_height, chunk_height * image_width);
    
  // Calculate send counts and displacements to scatter
  int *sendcounts = malloc(sizeof(int)*num_proc);		// Array describing how many elements to send to each process
  int *displs = malloc(sizeof(int)*num_proc);    	 	// Array describing the displacements where each segment begins
  int rem = image_height % num_proc; 					// Rows remaining after division among processes
  int sum = 0;                							// Sum of counts. Used to calculate displacements
  for (int i = 0; i < num_proc; i++) {
	if (i == 0 || i == num_proc - 1){
	  sendcounts[i] = (image_height / num_proc) + 1; 	// Number of rows per process (1 row extra for bottom and top chunks)
	  }
	else{
	  sendcounts[i] = (image_height / num_proc) + 2; 	// Number of rows per process (2 row extra for intermediate chunks)
	}
    if (rem > 0) {
      sendcounts[i] += 1; 								// If there are remaining rows, distribute them among the processes
      rem--;
    }	
	sendcounts[i] = sendcounts[i] * image_width; 		// Convert rows into image buffer elements
    displs[i] = sum;	
    sum += sendcounts[i] - 2 * image_width;				// Set displacement for next chunk
    printf("For process %d. Sendcounts(%d) = %d and displs(%d) = %d\n", my_rank, i, sendcounts[i], i, displs[i]);
  }
  
  // Scatter image between the processes
  MPI_Scatterv(data_1D, 			// Data on root process
	  		   sendcounts, 			// Array with the number of elements sent to each process
			   displs, 				// Displacement relative to the image buffer
			   MPI_UNSIGNED_CHAR, 	// Data type
			   subset,				// Where to place the scattered data
			   chunk_size, 			// Number of elements to receive
			   MPI_UNSIGNED_CHAR,  	// Data type
			   0, 					// Rank of root process
			   MPI_COMM_WORLD);		// MPI communicator
   printf("Process %d scattered succesfully\n", my_rank);
  
	
  // Convert 1D subset into a 2D subset to use the image kernel
  localImageChannel->height = sendcounts[my_rank] / image_width;
  localImageChannel->width = image_width;
  printf("My rank %d and my height %d\n",my_rank, localImageChannel->height);

  unsigned char **subset_2d;
  subset_2d = calloc(localImageChannel->height, sizeof(unsigned char *));
	
  for (unsigned int i = 0; i < localImageChannel->height; i++) {
    subset_2d[i] = calloc(localImageChannel->width, sizeof(unsigned char));
  }
	
  for (unsigned int i = 0; i < localImageChannel->height; i++) {
    for (unsigned int j = 0; j < localImageChannel->width; j++) {
      subset_2d[i][j] = subset[(i * localImageChannel->width) + j];
    }
  }
  printf("2D data generated succesfully\n");
  localImageChannel->data = subset_2d; 
  if (my_rank == 0) {printf("Check if scatter value is fine -- Original = %d\n", imageChannel->data[666][666]);}
  printf("My rank is %d. Check if scatter value is fine -- Subset = %d\n", my_rank, localImageChannel->data[666][666]);
  
  // Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x])
  bmpImageChannel *localProcessImageChannel = newBmpImageChannel(localImageChannel->width, localImageChannel->height);
  for (unsigned int i = 0; i < iterations; i ++) {
	applyKernel(localProcessImageChannel->data,
                localImageChannel->data,
                localImageChannel->width,
                localImageChannel->height,
                (int *)laplacian1Kernel, 3, laplacian1KernelFactor
 //               (int *)laplacian2Kernel, 3, laplacian2KernelFactor
 //               (int *)laplacian3Kernel, 3, laplacian3KernelFactor
 //               (int *)gaussianKernel, 5, gaussianKernelFactor
                );
    swapImageChannel(&localProcessImageChannel, &localImageChannel);
  }
  freeBmpImageChannel(localProcessImageChannel);
  printf ("Computation done on node %d\n",my_rank);
  
  printf("\n Prior to readjust. My rank is %d. Sendcounts = %d. Displs = %d\n",my_rank, sendcounts[my_rank]/image_width, displs[my_rank]/image_width);
  // Readjust displacements and sendcounts to gather properly
  int *recvcounts = malloc(sizeof(int)*num_proc);		// Array describing how many elements are received from each process
  for (int i = 0; i < num_proc; i++) {
    if ( displs[i] != 0 ){
	  displs[i] += image_width; 						// Revert displacement of lower halo
	}
	if ( i == 0 || i == num_proc - 1){
	  //sendcounts[i] -= image_width; 					// Send one row less for first and last chunk
	  recvcounts[i] = sendcounts[i] - image_width; 		// Send one row less for first and last chunk
	}
	else{
	  //sendcounts[i] -= 2*image_width; 	
	  recvcounts[i] = sendcounts[i] - 2*image_width;	// Send two rows less for intermediate chunks
	}
  }
  
  // Processed image channel to use gather function
  //bmpImageChannel *processImageChannel = newBmpImageChannel(0, 0);
  bmpImageChannel *processImageChannel = newBmpImageChannel(image_width, image_height);
  /*if (my_rank == 0) {
    processImageChannel = newBmpImageChannel(image_width, image_height);
  }*/
  
  // Local data in 1D tp feed the gather function
  localData_1D = (unsigned char*) calloc(localImageChannel->width * localImageChannel->height, sizeof(unsigned char));
  for (unsigned int i = 0; i < localImageChannel->height; i++) {
  for (unsigned int j = 0; j < localImageChannel->width; j++) {
    localData_1D[(i * localImageChannel->width) + j] = localImageChannel->data[i][j];
    }
  }
  
  unsigned char *gathered_buffer = calloc(image_width*image_height, sizeof(unsigned char));
  printf("\n Prior to gather. My rank is %d. Recvcounts = %d. Sendcounts = %d. Displs = %d\n",my_rank, recvcounts[my_rank]/image_width, sendcounts[my_rank]/image_width, displs[my_rank]/image_width);
  // Gather image chunks from the processes
  printf("Local data test %d\n", localImageChannel->data[30][87]);
  //if (my_rank == 0) {printf("Process data test %d\n", processImageChannel->data[0][image_width+1]);}
  MPI_Gatherv(localData_1D,					// Data to gather
			  recvcounts[my_rank],			// Number of elements of each gathered data
			  MPI_UNSIGNED_CHAR, 			// Data type
			  gathered_buffer,			 	// Where to place gathered data
			  recvcounts, 					// Number of elements received per process
		      displs,			  			// Displacement relative to the image buffer
			  MPI_UNSIGNED_CHAR,			// Data type
			  0,							// Rank of root process
			  MPI_COMM_WORLD);				// MPI communicator
  printf("Process %d gathered succesfully\n", my_rank);
	
  unsigned char **gathered_buffer_2d;
  gathered_buffer_2d = calloc(image_height, sizeof(unsigned char *));
	
  for (unsigned int i = 0; i < image_height; i++) {
    gathered_buffer_2d[i] = calloc(image_width, sizeof(unsigned char));
  }
	
  for (unsigned int i = 0; i < image_height; i++){
    for (unsigned int j = 0; j < image_width; j++) {
      gathered_buffer_2d[i][j] = gathered_buffer[i * image_width + j];
    }
  }  
  printf("2D data generated succesfully\n");
  processImageChannel->data = gathered_buffer_2d; 
  
   //if (my_rank == 0) {printf("Process data test %d\n", processImageChannel->data[0][image_width+5]);}

	free(recvcounts);
	free(sendcounts);
	free(displs);

  // Root proceess writes the image back to disk
  if (my_rank == 0) {
    // Map our single color image back to a normal BMP image with 3 color channels
    // mapEqual puts the color value on all three channels the same way
    // other mapping functions are mapRed, mapGreen, mapBlue
    bmpImage *image = newBmpImage(image_width,image_height);
    printf("Image dimemnsions. H = %d. W = %d\n",image->height, image->width); 
    printf("Processes dimemnsions. H = %d. W = %d\n",processImageChannel->height, processImageChannel->width); 
    
    if (mapImageChannel(image, processImageChannel, mapEqual) != 0) {
      fprintf(stderr, "Could not map image channel!\n");
      freeBmpImage(image);
      freeBmpImageChannel(processImageChannel);
      goto error_exit;
    }
    freeBmpImageChannel(processImageChannel);
   
    if (saveBmpImage(image, output) != 0) {
	  fprintf(stderr, "Could not save output to '%s'!\n", output);
	  freeBmpImage(image);
	  goto error_exit;
    }
  }
  
graceful_exit:
  ret = 0;
  
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);
    
  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return ret;
};
