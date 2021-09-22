/******************************************
 * smearing_techniques.cpp
 *
 * originally from
 *   smearing_techniques.cc
 *   Author: Marc Wagner
 *   Date: September 2007
 *
 * February 2010
 * ported to cvc_parallel, parallelized
 *
 * Fri Dec  9 09:28:18 CET 2016
 * ported to cvc
 ******************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#ifdef HAVE_MPI
#include <mpi.h>  
#endif
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "mpi_init.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "smearing_techniques.h"

namespace cvc {

/************************************************************
 * Performs a number of APE smearing steps
 *
 ************************************************************/
int APE_Smearing(double *smeared_gauge_field, double const APE_smearing_alpha, int const APE_smearing_niter) {

#if ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA )
  /***********************************************************
   * call  library function wrapper
   ***********************************************************/
  _performAPEnStep ( APE_smearing_niter, APE_smearing_alpha );

#else

  if ( APE_smearing_niter <= 0 ) return(0);

  const unsigned int gf_items = 72*VOLUME;
  const size_t gf_bytes = gf_items * sizeof(double);

  int iter;
  double *smeared_gauge_field_old = NULL;
  alloc_gauge_field(&smeared_gauge_field_old, VOLUMEPLUSRAND);

  for(iter=0; iter<APE_smearing_niter; iter++) {

    memcpy((void*)smeared_gauge_field_old, (void*)smeared_gauge_field, gf_bytes);
#ifdef HAVE_MPI
    xchange_gauge_field(smeared_gauge_field_old);
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif

    double M1[18], M2[18];
    unsigned int index;
    unsigned int index_mx_1, index_mx_2, index_mx_3;
    unsigned int index_px_1, index_px_2, index_px_3;
    unsigned int index_my_1, index_my_2, index_my_3;
    unsigned int index_py_1, index_py_2, index_py_3;
    unsigned int index_mz_1, index_mz_2, index_mz_3;
    unsigned int index_pz_1, index_pz_2, index_pz_3;
    double *U = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(unsigned int idx = 0; idx < VOLUME; idx++) {
  
      /************************
       * Links in x-direction.
       ************************/
      index = _GGI(idx, 1);
  
      index_my_1 = _GGI(g_idn[idx][2], 2);
      index_my_2 = _GGI(g_idn[idx][2], 1);
      index_my_3 = _GGI(g_idn[g_iup[idx][1]][2], 2);
  
      index_py_1 = _GGI(idx, 2);
      index_py_2 = _GGI(g_iup[idx][2], 1);
      index_py_3 = _GGI(g_iup[idx][1], 2);
  
      index_mz_1 = _GGI(g_idn[idx][3], 3);
      index_mz_2 = _GGI(g_idn[idx][3], 1);
      index_mz_3 = _GGI(g_idn[g_iup[idx][1]][3], 3);
  
      index_pz_1 = _GGI(idx, 3);
      index_pz_2 = _GGI(g_iup[idx][3], 1);
      index_pz_3 = _GGI(g_iup[idx][1], 3);
  
  
      U = smeared_gauge_field + index;
      _cm_eq_zero(U);
  
      /* negative y-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
  
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive y-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
  
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* negative z-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
  
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive z-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
  
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      /* center */
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index);
  
      /* Projection to SU(3). */
      cm_proj(U);
  
  
      /***********************
       * Links in y-direction.
       ***********************/
  
      index = _GGI(idx, 2);
  
      index_mx_1 = _GGI(g_idn[idx][1], 1);
      index_mx_2 = _GGI(g_idn[idx][1], 2);
      index_mx_3 = _GGI(g_idn[g_iup[idx][2]][1], 1);
  
      index_px_1 = _GGI(idx, 1);
      index_px_2 = _GGI(g_iup[idx][1], 2);
      index_px_3 = _GGI(g_iup[idx][2], 1);
  
      index_mz_1 = _GGI(g_idn[idx][3], 3);
      index_mz_2 = _GGI(g_idn[idx][3], 2);
      index_mz_3 = _GGI(g_idn[g_iup[idx][2]][3], 3);
  
      index_pz_1 = _GGI(idx, 3);
      index_pz_2 = _GGI(g_iup[idx][3], 2);
      index_pz_3 = _GGI(g_iup[idx][2], 3);
  
      U = smeared_gauge_field + index;
      _cm_eq_zero(U);
  
      /* negative x-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive x-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* negative z-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mz_2, smeared_gauge_field_old + index_mz_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive z-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_pz_2, smeared_gauge_field_old + index_pz_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_pz_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      /* center */
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index);
  
      /* Projection to SU(3). */
      cm_proj(U);
  
      /**************************
       * Links in z-direction.
       **************************/
  
      index = _GGI(idx, 3);
  
      index_mx_1 = _GGI(g_idn[idx][1], 1);
      index_mx_2 = _GGI(g_idn[idx][1], 3);
      index_mx_3 = _GGI(g_idn[g_iup[idx][3]][1], 1);
  
      index_px_1 = _GGI(idx, 1);
      index_px_2 = _GGI(g_iup[idx][1], 3);
      index_px_3 = _GGI(g_iup[idx][3], 1);
  
      index_my_1 = _GGI(g_idn[idx][2], 2);
      index_my_2 = _GGI(g_idn[idx][2], 3);
      index_my_3 = _GGI(g_idn[g_iup[idx][3]][2], 2);
  
      index_py_1 = _GGI(idx, 2);
      index_py_2 = _GGI(g_iup[idx][2], 3);
      index_py_3 = _GGI(g_iup[idx][3], 2);
  
      U = smeared_gauge_field + index;
      _cm_eq_zero(U);
  
      /* negative x-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_mx_2, smeared_gauge_field_old + index_mx_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_mx_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive x-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_px_2, smeared_gauge_field_old + index_px_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_px_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* negative y-direction */
      _cm_eq_cm_ti_cm(M1, smeared_gauge_field_old + index_my_2, smeared_gauge_field_old + index_my_3);
      _cm_eq_cm_dag_ti_cm(M2, smeared_gauge_field_old + index_my_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      /* positive y-direction */
      _cm_eq_cm_ti_cm_dag(M1, smeared_gauge_field_old + index_py_2, smeared_gauge_field_old + index_py_3);
      _cm_eq_cm_ti_cm(M2, smeared_gauge_field_old + index_py_1, M1);
      _cm_pl_eq_cm(U, M2);
  
      _cm_ti_eq_re(U, APE_smearing_alpha);
  
      /* center */
      _cm_pl_eq_cm(U, smeared_gauge_field_old + index);
  
      /* Projection to SU(3). */
      cm_proj(U);
    }  /* end of loop on ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on number of smearing steps */

  free(smeared_gauge_field_old);
#ifdef HAVE_MPI
  xchange_gauge_field(smeared_gauge_field);
#endif

#endif  /* of if ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA ) */
  return(0);
}  /* end of APE_Smearing */


/*****************************************************
 *
 * Performs a number of Jacobi smearing steps
 *
 * smeared_gauge_field  = gauge field used for smearing (in)
 * psi = quark spinor (in/out)
 * N, kappa = Jacobi smearing parameters (in)
 *
 *****************************************************/
int Jacobi_Smearing(double *smeared_gauge_field, double *psi, int const N, double const kappa) {

  if ( N == 0 || kappa == 0.0 ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [Jacobi_Smearing] Warning, nothing to smear\n");
    return ( 0 );
  }

#if ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA )
  _performWuppertalnStep ( psi, psi, N, kappa );
#else

  const size_t sf_items = _GSI(VOLUME);
  const size_t sf_bytes = sf_items * sizeof(double);
  const double norm = 1.0 / (1.0 + 6.0*kappa);

  int i1;
  double *psi_old = (double*)malloc( _GSI(VOLUME+RAND)*sizeof(double));
  if(psi_old == NULL) {
    fprintf(stderr, "[Jacobi_Smearing] Error from malloc\n");
    return(3);
  }

  if ( smeared_gauge_field == NULL || psi == NULL ) {
    fprintf(stderr, "[Jacobi_Smearing] Error, input / output fields are NULL \n");
    return(4);
  }

#ifdef HAVE_MPI
  xchange_gauge_field (smeared_gauge_field);
#endif

  /* loop on smearing steps */
  for(i1 = 0; i1 < N; i1++) {

    memcpy((void*)psi_old, (void*)psi, sf_bytes);
#ifdef HAVE_MPI
    xchange_field(psi_old);
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif

    unsigned int idx, idy;
    unsigned int index_s, index_s_mx, index_s_px, index_s_my, index_s_py, index_s_mz, index_s_pz, index_g_mx;
    unsigned int index_g_px, index_g_my, index_g_py, index_g_mz, index_g_pz; 
    double *s=NULL, spinor[24];

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(idx = 0; idx < VOLUME; idx++) {
      index_s = _GSI(idx);

      index_s_mx = _GSI(g_idn[idx][1]);
      index_s_px = _GSI(g_iup[idx][1]);
      index_s_my = _GSI(g_idn[idx][2]);
      index_s_py = _GSI(g_iup[idx][2]);
      index_s_mz = _GSI(g_idn[idx][3]);
      index_s_pz = _GSI(g_iup[idx][3]);

      idy = idx;
      index_g_mx = _GGI(g_idn[idy][1], 1);
      index_g_px = _GGI(idy, 1);
      index_g_my = _GGI(g_idn[idy][2], 2);
      index_g_py = _GGI(idy, 2);
      index_g_mz = _GGI(g_idn[idy][3], 3);
      index_g_pz = _GGI(idy, 3);

      s = psi + _GSI(idy);
      _fv_eq_zero(s);

      /* negative x-direction */
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mx, psi_old + index_s_mx);
      _fv_pl_eq_fv(s, spinor);

      /* positive x-direction */
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_px, psi_old + index_s_px);
      _fv_pl_eq_fv(s, spinor);

      /* negative y-direction */
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_my, psi_old + index_s_my);
      _fv_pl_eq_fv(s, spinor);

      /* positive y-direction */
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_py, psi_old + index_s_py);
      _fv_pl_eq_fv(s, spinor);

      /* negative z-direction */
      _fv_eq_cm_dag_ti_fv(spinor, smeared_gauge_field + index_g_mz, psi_old + index_s_mz);
      _fv_pl_eq_fv(s, spinor);

      /* positive z-direction */
      _fv_eq_cm_ti_fv(spinor, smeared_gauge_field + index_g_pz, psi_old + index_s_pz);
      _fv_pl_eq_fv(s, spinor);


      /* Put everything together; normalization. */
      _fv_ti_eq_re(s, kappa);
      _fv_pl_eq_fv(s, psi_old + index_s);
      _fv_ti_eq_re(s, norm);
    }  /* end of loop on ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on smearing steps */

  free(psi_old);

#endif  /* of ( defined HAVE_TMLQCD_LIBWRAPPER ) && ( defined _SMEAR_QUDA ) */

  return(0);
}  /* end of Jacobi_Smearing */

/******************************************
 *
 ******************************************/
inline void _ADD_STAPLES_TO_COMPONENT( double * const buff_out, double * const buff_in, unsigned int const x,  int const to, int const via) 
{
  double tmp[18], tmp2[18];

  double * const _buff_in_up_via_to        = buff_in + _GGI( g_iup[x][via], to );
  double * const _buff_in_up_to_via        = buff_in + _GGI( g_iup[x][to], via );

  double * const _buff_in_via              = buff_in + _GGI( x, via);

  double * const _buff_in_dn_via_to        = buff_in + _GGI( g_idn[x][via], to);
  double * const _buff_in_dn_via_up_to_via = buff_in + _GGI( g_iup[g_idn[x][via]][to], via );
  double * const _buff_in_dn_via_via       = buff_in + _GGI( g_idn[x][via], via);

    _cm_eq_cm_ti_cm_dag ( tmp, _buff_in_up_via_to, _buff_in_up_to_via );
    _cm_eq_cm_ti_cm ( tmp2, _buff_in_via, tmp);
    _cm_pl_eq_cm ( buff_out, tmp2 );
    _cm_eq_cm_ti_cm ( tmp, _buff_in_dn_via_to, _buff_in_dn_via_up_to_via );
    _cm_eq_cm_dag_ti_cm ( tmp2, _buff_in_dn_via_via, tmp );
    _cm_pl_eq_cm ( buff_out, tmp2 );
}  /* end of _ADD_STAPLES_TO_COMPONENT */

/******************************************
 *
 ******************************************/
void generic_staples ( double * const buff_out, const unsigned int x, const int mu, double * const buff_in )
{

  _cm_eq_zero ( buff_out );

  switch (mu)
  {
    case 0:
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 0, 1);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 0, 2);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 0, 3);
      break;

    case 1:
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 1, 0);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 1, 2);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 1, 3);
      break;

    case 2:
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 2, 0);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 2, 1);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 2, 3);
      break;

    case 3:
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 3, 0);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 3, 1);
      _ADD_STAPLES_TO_COMPONENT(buff_out, buff_in, x, 3, 2);
      break;
  }

}  /* end of generic_staples */


/******************************************
 *
 ******************************************/
inline void exposu3( double * const vr, double * const p ) {
  
  double v[18], v2[18];

#if 0
  /* it writes 'p=vec(h_{j,mu})' in matrix form 'v' */
  _make_su3(v,*p);
#endif
  _cm_eq_cm ( v, p );

  /* v2 = v^2 */
  _cm_eq_cm_ti_cm ( v2, v, v);

  /* 1/2 real part of trace of v2 */
  double const a = 0.5 * ( v2[0] + v2[8] + v2[16] );

  /* 1/3 imaginary part of tr v*v2 */
  double const b = 0.33333333333333333 * (
        v[ 0] * v2[ 1] + v[ 1] * v2[ 0]
      + v[ 2] * v2[ 7] + v[ 3] * v2[ 6]
      + v[ 4] * v2[13] + v[ 5] * v2[12]
      + v[ 6] * v2[ 3] + v[ 7] * v2[ 2]
      + v[ 8] * v2[ 9] + v[ 9] * v2[ 8]
      + v[10] * v2[15] + v[11] * v2[14]
      + v[12] * v2[ 5] + v[13] * v2[ 4]
      + v[14] * v2[11] + v[15] * v2[10]
      + v[16] * v2[17] + v[17] * v2[16] );

  double _Complex a0  = 0.16059043836821615e-9;   /* 1 / 13 ! */
  double _Complex a1  = 0.11470745597729725e-10;  /* 1 / 14 ! */
  double _Complex a2  = 0.76471637318198165e-12;  /* 1 / 15 ! */
  double fac = 0.20876756987868099e-8;            /* 1 / 12 ! */
  double r   = 12.0;
  _Complex double a1p;

  for ( int i = 3; i <= 15; ++i)
  {
    a1p = a0 + a * a2;
    a0 = fac + b * I * a2;
    a2 = a1;
    a1 = a1p;
    fac *= r;
    r -= 1.0;
  }

  /* vr = a0 + a1*v + a2*v2 */
  _cm_eq_zero( vr );

  /* vr = a0 * diag ( 1,1,1 ) */
  vr[ 0] = creal( a0 );
  vr[ 1] = cimag( a0 );
  vr[ 8] = creal( a0 );
  vr[ 9] = cimag( a0 );
  vr[16] = creal( a0 );
  vr[17] = cimag( a0 );

  double _Complex z;
  for ( int i = 0; i < 9; i++ ) {
    z = ( v[2*i] + v[2*i+1] * I ) * a1 + ( v2[2*i] + v2[2*i+1] * I ) * a2;
    vr[2*i]   += creal( z );
    vr[2*i+1] += cimag( z );
  }

}  /* end of exposu3 */


/******************************************
 *
 ******************************************/
int stout_smear_inplace ( double * const m_field, const int stout_n, const double stout_rho, double * const buffer )
{
  
  /* start of the the stout smearing **/
  for ( int iter = 0; iter < stout_n; ++iter)
  {
#ifdef HAVE_MPI
    xchange_gauge_field ( m_field );
#endif
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
      double tmp[18];
      double * _buffer = NULL;
      double * _m_field = NULL;
      unsigned int iix;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for ( unsigned int x = 0; x < VOLUME; ++x) {
      for ( int mu = 0; mu < 4; ++mu)
      {
        iix      = _GGI( x, mu);
        _buffer  = buffer + iix;
        _m_field = m_field + iix;

        generic_staples( tmp, x, mu, m_field );
        _cm_ti_eq_re( tmp, stout_rho );
        _cm_eq_cm_ti_cm_dag ( _buffer, tmp, _m_field );
        _cm_eq_antiherm_trless_cm ( tmp, _buffer );
        exposu3 ( _buffer, tmp );
      }
    }

#ifdef HAVE_OPENMP
#pragma omp barrier
#pragma omp for
#endif
    for ( unsigned int x = 0; x < VOLUME; ++x) {
      for(int mu = 0 ; mu < 4; ++mu)
      {
        iix      = _GGI( x, mu);
        _buffer  = buffer + iix;
        _m_field = m_field + iix;

        _cm_eq_cm_ti_cm( tmp, _buffer, _m_field );
        _cm_eq_cm ( _m_field, tmp );
      }
    }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  }  /* end of loop on iterations */

  return(0);
}  /* end of stout_smear_inplace */

/******************************************/
/******************************************/

}  /* end of namespace cvc */
