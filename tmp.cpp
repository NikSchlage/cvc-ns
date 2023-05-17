/************************************************************************************/
/************************************************************************************/

#ifdef HAVE_LIBLEMON
/************************************************************************************
 * read lemon binary propagator
 ************************************************************************************/
int read_binary_propagator_data(double * const s, LemonReader * reader, const int prec, DML_Checksum *checksum) {

  n_uint64_t const real_block_size = 288;

  int  status = 0;
  int latticeSize[] = {T_global, LX_global, LY_global, LZ_global};
  int scidacMapping[] = {0, 3, 2, 1};
  n_uint64_t bytes;
  double *p = NULL;
  char *filebuffer = NULL, *current = NULL;
  double tick = 0, tock = 0;
  DML_SiteRank rank;
  uint64_t fbspin;
  char measure[64];
  int words_bigendian = big_endian();

  DML_checksum_init(checksum);

  fbspin = real_block_size * sizeof(double);
  if (prec == 32) fbspin /= 2;
  bytes = fbspin;

  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    fprintf (stderr, "[read_binary_propagator_data] malloc errno in read_binary_spinor_data_parallel %s %d\n", __FILE__, __LINE__);
    EXIT(501);
  }

  status = lemonReadLatticeParallelMapped(reader, filebuffer, bytes, latticeSize, scidacMapping);

  if (status < 0 && status != LEMON_EOR) {
    fprintf(stderr, "[read_binary_propagator_data] LEMON read error occured with status = %d while reading!\nPanic! Aborting... %s %d\n",
        status, __FILE__, __LINE__);
    MPI_File_close(reader->fp);
    EXIT(502);
  }

  for ( int t = 0; t <  T; t++) {
  for ( int z = 0; z < LZ; z++) {
  for ( int y = 0; y < LY; y++) {
  for ( int x = 0; x < LX; x++) {
    rank = (DML_SiteRank)( LXstart + (((Tstart  + t) * (LZ*g_nproc_z) + LZstart + z) * (LY*g_nproc_y) + LYstart + y) * ((DML_SiteRank) LX * g_nproc_x) + x);
    current = filebuffer + bytes * (x + (y + (t * LZ + z) * LY) * LX);
    DML_checksum_accum(checksum, rank, current, bytes);

    unsigned int const i = g_ipt[t][x][y][z];
    p = s + real_block_size * i;
    if(!words_bigendian) {
      if (prec == 32)
        byte_swap_assign_single2double(p, current, real_block_size * sizeof(double) / 8);
      else
        byte_swap_assign(p, current, real_block_size * sizeof(double) / 8);
    } else {
      if (prec == 32)
        single2double(p, current, real_block_size * sizeof(double) / 8);
      else
        memcpy(p, current, real_block_size * sizeof(double));
    }
  }}}}

  DML_global_xor(&checksum->suma);
  DML_global_xor(&checksum->sumb);

  free(filebuffer);
  return(0);
}  /* end of read_binary_propagator_data */

/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * read propagator using lemon
 ************************************************************************************/
int read_lime_propagator(double * const s, char * filename, const int position) {

  n_uint64_t const real_block_size = 288;

  MPI_File *ifs;
  int status = 0, getpos = 0, prec = 0, prop_type;
  char *header_type = NULL;
  LemonReader *reader = NULL;
  DML_Checksum checksum;
  n_uint64_t bytes = 0;

  if(g_cart_id==0)
    fprintf(stdout, "# [read_lime_propagator] reading prop in LEMON format from file %s at pos %d\n", filename, position);

  ifs = (MPI_File*)malloc(sizeof(MPI_File));
  status = MPI_File_open(g_cart_grid, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, ifs);
  status = (status == MPI_SUCCESS) ? 0 : 1;
  if(status) {
    fprintf(stderr, "[read_lime_propagator] Err, could not open file for reading\n");
    EXIT(500);
  }
  
  if( (reader = lemonCreateReader(ifs, g_cart_grid))==NULL ) {
    fprintf(stderr, "[read_lime_propagator] Error, could not create lemon reader.\n");
    EXIT(502);
  }

  while ((status = lemonReaderNextRecord(reader)) != LIME_EOF) {
    if (status != LIME_SUCCESS) {
      fprintf(stderr, "[read_lime_propagator] lemonReaderNextRecord returned status %d.\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = (char*)lemonReaderType(reader);
    if (strcmp("scidac-binary-data", header_type) == 0) {
      if (getpos == position)
        break;
      else
        ++getpos;
    }
  }

  if (status == LIME_EOF) {
    fprintf(stderr, "[read_lime_propagator] Error, no scidac-binary-data record found in file.\n");
    EXIT(500);
  } 

  bytes = lemonReaderBytes(reader);
  if (bytes == (n_uint64_t)LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * T_global * real_block_size * sizeof(double)) {
    prec = 64;
  } else {
    if (bytes == (n_uint64_t)LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * T_global * real_block_size * sizeof(double) / 2) {
      prec = 32;
    } else {
      if(g_cart_id==0) fprintf(stderr, "[read_lime_propagator] Error, wrong length in spinor. Aborting read!\n");
       EXIT(501);
    }
  }
  if(g_cart_id==0) fprintf(stdout, "# [read_lime_propagator] %d bit precision read.\n", prec);

  read_binary_propagator_data(s, reader, prec, &checksum);

  if (g_cart_id == 0) fprintf(stdout, "# [read_lime_propagator] checksum for DiracFermion field in file %s position %d is %#x %#x\n", 
    filename, position, checksum.suma, checksum.sumb);

  lemonDestroyReader(reader);
  MPI_File_close(ifs);
  free(ifs);
  
  return(0);
}  /* read_lime_propagator */

#else  /* ! LEMON but LIME */
/************************************************************************************
 * read lime binary propagator
 ************************************************************************************/
int read_binary_propagator_data(double * const s, LimeReader * limereader, const int prec, DML_Checksum *ans) {

  n_uint64_t const real_block_size = 288;

  int status=0;
  n_uint64_t bytes, ix;
  double tmp[real_block_size];
  float  tmp2[real_block_size];
  DML_SiteRank rank;
  int words_bigendian;
  int t, x, y, z;
  words_bigendian = big_endian();

  DML_checksum_init(ans);
  rank = (DML_SiteRank) 0;
  
  if ( prec == 32 ) bytes = real_block_size * sizeof(float);
  else              bytes = real_block_size * sizeof(double);
  for(t = 0; t < T; t++){
    for(z = 0; z < LZ; z++){
      for(y = 0; y < LY; y++){
#if (defined HAVE_MPI)
      limeReaderSeek(limereader,(n_uint64_t) ( (((Tstart+t)*(LZ*g_nproc_z) + LZstart + z)*(LY*g_nproc_y)+LYstart+y)*(LX*g_nproc_x) +LXstart )*bytes, SEEK_SET);
#endif
	for(x = 0; x < LX; x++){
	  ix = g_ipt[t][x][y][z] * real_block_size;
	  rank = (DML_SiteRank) ((((Tstart+t)*(LZ*g_nproc_z)+LZstart + z)*(LY*g_nproc_y) + LYstart + y)*(DML_SiteRank)(LX*g_nproc_x) + LXstart + x);
	  if(prec == 32) {
	    status = limeReaderReadData(tmp2, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp2, bytes);	    
	  }
	  else {
	    status = limeReaderReadData(tmp, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp, bytes);
	  }
	  if(!words_bigendian) {
	    if(prec == 32) {
	      byte_swap_assign_single2double(&s[ix], (float*)tmp2, (int)real_block_size);
	    }
	    else {
	      byte_swap_assign(&s[ix], tmp, (int)real_block_size);
	    }
	  }
	  else {
	    if(prec == 32) {
	      single2double(&s[ix], (float*)tmp2, (int)real_block_size );
	    }
	    else memcpy(&s[ix], tmp, bytes);
	  }
	  if(status < 0 && status != LIME_EOR) {
	    return(-1);
	  }
	}
      }
    }
  }
#ifdef HAVE_MPI
  DML_checksum_combine(ans);
#endif
  if(g_cart_id == 0) printf("# [read_binary_propagator_data] The final checksum is %#lx %#lx\n", (*ans).suma, (*ans).sumb);
  return(0);
}  /* end of read_binary_propagator_data */

/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * read lime propagator
 ************************************************************************************/
int read_lime_propagator(double * const s, char * filename, const int position) {

  uint64_t const real_block_size = 288;

  FILE * ifs;
  int status=0, getpos=-1;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  n_uint64_t prec = 32;
  DML_Checksum checksum;
  
  if((ifs = fopen(filename, "r")) == (FILE*)NULL) {
    fprintf(stderr, "[read_lime_propagator] Error opening file %s\n", filename);
    return(-1);
  }
  if(g_proc_id==0) fprintf(stdout, "# [read_lime_propagator] Reading Dirac-fermion field in LIME format from %s\n", filename);

  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_propagator] Unable to open LimeReader\n");
    return(-1);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_propagator] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("scidac-binary-data",header_type) == 0) getpos++;
    if(getpos == position) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_propagator] no scidac-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    return(-2);
  }
  bytes = limeReaderBytes(limereader);
  if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(real_block_size*sizeof(double))) prec = 64;
  else if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(real_block_size*sizeof(float))) prec = 32;
  else {
    fprintf(stderr, "[read_lime_propagator] wrong length in eoprop: bytes = %lu, not %lu. Aborting read!\n", 
	    bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(real_block_size*sizeof(double)));
    return(-1);
  }
  if(g_cart_id == 0) printf("# [read_lime_propagator] %llu Bit precision read\n", prec);

  status = read_binary_propagator_data(s, limereader, prec, &checksum);

  if(status < 0) {
    fprintf(stderr, "[read_lime_propagator] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
	    status, filename);
    EXIT(500);
  }

  limeDestroyReader(limereader);
  fclose(ifs);
  return(0);
}  /* end of read_lime_propagator  */
#endif  /* HAVE_LIBLEMON */
