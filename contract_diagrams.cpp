/****************************************************
 * contract_diagrams.c
 * 
 * Mon Jun  5 16:00:53 CDT 2017
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "gamma.h"
#include "zm4x4.h"
#include "contract_diagrams.h"

namespace cvc {

#if 0
/****************************************************
 * we always sum in the following way
 * v2[alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, beta]  v3[beta,m]
 ****************************************************/
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[3], unsigned int N, int init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {

    for ( int alpha = 0; alpha < 4; alpha++ ) {
      for ( int beta = 0; beta < 4; beta++ ) {

        int vdiag_index = 4 * alpha + beta;
        /* vdiag[it][vdiag_index] = 0.; */

        /****************************************************/
        /****************************************************/

        for ( int gamma = 0; gamma < 4; gamma++ ) {

          int idx[3] = { alpha, beta, gamma };

          int pidx[3] = { idx[perm[0]], idx[perm[1]], idx[perm[2]] };

          for ( int delta = 0; delta < 4; delta++ ) {
            for ( int m = 0; m < 3; m++ ) {

              /* use the permutation */
              int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
              int v3_index  = 3 * delta + m;

              vdiag[it][vdiag_index] -=  v2[it][v2_pindex] * v3[it][v3_index] * g.m[gamma][delta];
            }  /* end of loop on color index m */
          }  /* end of loop on spin index delta */
        }  /* end of loop on spin index gamma */

        /****************************************************/
        /****************************************************/

      }  /* end of loop on spin index beta */
    }  /* end of loop on spin index alpha */

  }  /* end of loop on N */

}  /* end of function contract_diagram_v2_gamma_v3 */
#endif  /* of if 0*/

/****************************************************
 * we always sum in the following way
 * v2[alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, alpha_3]  v3[ alpha_p[3], m]
 ****************************************************/
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[4], unsigned int N, int init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [contract_diagram_v2_gamma_v3] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {

    for ( int alpha = 0; alpha < 4; alpha++ ) {
      for ( int beta = 0; beta < 4; beta++ ) {

        int vdiag_index = 4 * alpha + beta;
        /* vdiag[it][vdiag_index] = 0.; */

        /****************************************************/
        /****************************************************/

        for ( int gamma = 0; gamma < 4; gamma++ ) {
          for ( int delta = 0; delta < 4; delta++ ) {

            int idx[4]  = { alpha, beta, gamma, delta };

            int pidx[4] = { idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]] };

            for ( int m = 0; m < 3; m++ ) {

              /* use the permutation */
              int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
              int v3_index  = 3 * pidx[3] + m;

              vdiag[it][vdiag_index] -=  v2[it][v2_pindex] * v3[it][v3_index] * g.m[gamma][delta];
            }  /* end of loop on color index m */
          }  /* end of loop on spin index delta */
        }  /* end of loop on spin index gamma */

        /****************************************************/
        /****************************************************/

      }  /* end of loop on spin index beta */
    }  /* end of loop on spin index alpha */

  }  /* end of loop on N */

  return(0);
}  /* end of function contract_diagram_v2_gamma_v3 */

/****************************************************
 * we always sum in the following way
 * goet[b_oet][a_oet]  v2[a_oet][alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, alpha_3]  v3[b_oet][ alpha_p[3], m]
 ****************************************************/
int contract_diagram_oet_v2_gamma_v3 ( double _Complex **vdiag, double _Complex ***v2, double _Complex ***v3, gamma_matrix_type goet, gamma_matrix_type g, int perm[4], unsigned int N, int init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [contract_diagram_oet_v2_amma_v3] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

  for ( int sigma_oet = 0; sigma_oet < 4; sigma_oet++ ) {
  for ( int tau_oet   = 0; tau_oet   < 4; tau_oet++ ) {

    double _Complex c = goet.m[tau_oet][sigma_oet];
    if ( c == 0 ) continue;

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int it = 0; it < N; it++ ) {

      for ( int alpha = 0; alpha < 4; alpha++ ) {
        for ( int beta = 0; beta < 4; beta++ ) {

          int vdiag_index = 4 * alpha + beta;
          /* vdiag[it][vdiag_index] = 0.; */

          /****************************************************/
          /****************************************************/

          for ( int gamma = 0; gamma < 4; gamma++ ) {
            for ( int delta = 0; delta < 4; delta++ ) {

              int idx[4]  = { alpha, beta, gamma, delta };

              int pidx[4] = { idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]] };

              for ( int m = 0; m < 3; m++ ) {

                /* use the permutation */
                int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
                int v3_index  = 3 * pidx[3] + m;

                vdiag[it][vdiag_index] -=  c * v2[sigma_oet][it][v2_pindex] * v3[tau_oet][it][v3_index] * g.m[gamma][delta];
              }  /* end of loop on color index m */
            }  /* end of loop on spin index delta */
          }  /* end of loop on spin index gamma */

          /****************************************************/
          /****************************************************/

        }  /* end of loop on spin index beta */
      }  /* end of loop on spin index alpha */

    }  /* end of loop on N */

  }  /* end of loop on tau   oet */
  }  /* end of loop on sigma oet */
  return(0);
}  /* end of function contract_diagram_oet_v2_gamma_v3 */

#if 0
/****************************************************
 *
 ****************************************************/
void contract_b1 (double _Complex ***b1, double _Complex **v3, **double v2, gamma_matrix_type g) {

  for( int it = 0; it < T; it++ ) {
    for(int alpha = 0; alpha < 4; alpha++) {
    for(int beta = 0; beta < 4; beta++) {
      double _Complex z;
      for(int m = 0; m < 3; m++) {
        for(int gamma = 0; gamma < 4; gamma++) {
        for(int delta = 0; delta < 4; delta++) {
          int i3 = 3*gamma + m;
          int i2 = 4 * ( 4 * ( 4*m + beta ) + alpha ) + delta;
          z += -v3[it][i3] * v2[it][i2] * g.m[gamma][delta];
        }}
      }
      b1[it][alpha][beta] = z;
    }}
  }  /* loop on timeslices */
}  /* end of contract_b1 */

void contract_b2 (double _Complex ***b2, double _Complex **v3, **double v2, gamma_matrix_type g) {

  for( int it = 0; it < T; it++ ) {
    for(int alpha = 0; alpha < 4; alpha++) {
    for(int beta = 0; beta < 4; beta++) {
      double _Complex z;
      for(int m = 0; m < 3; m++) {
        for(int gamma = 0; gamma < 4; gamma++) {
        for(int delta = 0; delta < 4; delta++) {
          int i3 = 3*gamma + m;
          int i2 = 4 * ( 4 * ( 4*m + delta ) + alpha ) + beta;
          z += -v3[it][i3] * v2[it][i2] * g.m[gamma][delta];
        }}
      }
      b2[it][alpha][beta] = z;
    }}
  }  /* loop on timeslices */
}  /* end of contract_b2 */
#endif  /* end of if 0 */

/****************************************************
 * search for m1 in m2
 ****************************************************/
int match_momentum_id ( int **pid, int **m1, int **m2, int N1, int N2 ) {
#if 0 
  fprintf(stdout, "# [match_momentum_id] N1 = %d N2 = %d m2 == NULL ? %d\n", N1, N2 , m2 == NULL);
  for ( int i = 0; i < N1; i++ ) {
    fprintf(stdout, "# [match_momentum_id] m1 %d  %3d %3d %3d\n", i, m1[i][0], m1[i][1], m1[i][2]);
  }

  for ( int i = 0; i < N2; i++ ) {
    fprintf(stdout, "# [match_momentum_id] m2 %d  %3d %3d %3d\n", i, m2[i][0], m2[i][1], m2[i][2]);
  }
  return(1);
#endif

  if ( N1 > N2 ) {
    fprintf(stderr, "[match_momentum_id] Error, N1 > N2\n");
    return(1);
  }

  if ( *pid == NULL ) {
    *pid = (int*)malloc (N1 * sizeof(int) );
  }

  for ( int i = 0; i < N1; i++ ) {
    int found = 0;
    int p[3] = { m1[i][0], m1[i][1], m1[i][2] };

    for ( int k = 0; k < N2; k++ ) {
      if ( p[0] == m2[k][0] && p[1] == m2[k][1] && p[2] == m2[k][2] ) {
        (*pid)[i] = k;
        found = 1;
        break;
      }
    }
    if ( found == 0 ) {
      fprintf(stderr, "[match_momentum_id] Warning, could not find momentum no %d = %3d %3d %3d\n",
          i, p[0], p[1], p[2]);
      (*pid)[i] = -1;
      /* return(2); */
    }
  }

  /* TEST */
  if ( g_verbose > 2 ) {
    for ( int i = 0; i < N1; i++ ) {
      fprintf(stdout, "# [match_momentum_id] m1[%2d] = %3d %3d %3d matches m2[%2d] = %3d %3d %3d\n",
          i, m1[i][0], m1[i][1], m1[i][2],
          (*pid)[i], m2[(*pid)[i]][0], m2[(*pid)[i]][1], m2[(*pid)[i]][2]);
    }
  }

  return(0);
}  /* end of match_momentum_id */

/***********************************************
 * multiply x-space spinor propagator field
 *   with boundary phase
 ***********************************************/
int correlator_add_baryon_boundary_phase ( double _Complex ***sp, int tsrc) {

  if( g_propagator_bc_type == 0 ) {
    /* multiply with phase factor */
    fprintf(stdout, "# [correlator_add_baryon_boundary_phase] multiplying with boundary phase factor\n");

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for( int it = 0; it < T; it++ ) {
      int ir = (it + g_proc_coords[0] * T - tsrc + T_global) % T_global;
      const double _Complex w = cexp ( 3. * M_PI*(double)ir / (double)T_global  );
      zm4x4_ti_eq_co ( sp[it], w );
    }

  } else if ( g_propagator_bc_type == 1 ) {
    /* multiply with step function */
    fprintf(stdout, "# [add_baryon_boundary_phase] multiplying with boundary step function\n");

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for( int ir = 0; ir < T; ir++) {
      int it = ir + g_proc_coords[0] * T;  /* global t-value, 0 <= t < T_global */
      if(it < tsrc) {
        zm4x4_ti_eq_re ( sp[it], -1. );
      }  /* end of if it < tsrc */
    }  /* end of loop on ir */
  }

  return(0);
}  /* end of correlator_add_baryon_boundary_phase */


/***********************************************
 * multiply with phase from source location
 * - using pi1 + pi2 = - ( pf1 + pf2 ), so
 *   pi1 = - ( pi2 + pf1 + pf2 )
 ***********************************************/
int correlator_add_source_phase ( double _Complex ***sp, int p[3], int source_coords[3], unsigned int N ) {

  const double TWO_MPI = 2. * M_PI;

  const double _Complex w = cexp ( TWO_MPI * ( ( p[0] / (double)LX_global ) * source_coords[0] + 
                                               ( p[1] / (double)LY_global ) * source_coords[1] + 
                                               ( p[2] / (double)LZ_global ) * source_coords[2] ) );
  
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) {
    zm4x4_ti_eq_co ( sp[ix], w );
  }
  return(0);
}  /* end of correlator_add_source_phase */

int correlator_spin_projection (double _Complex ***sp_out, double _Complex ***sp_in, int i, int k, double a, double b, unsigned N) {

  int ik = 4*i+k;
  
  switch(ik) {
    case  5: 
    case 10: 
    case 15: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_33 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case  6: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_12 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case  7: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_13 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case  9: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_21 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case 11: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_23 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case 13: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_31 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    case 14: 
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for( unsigned int ir = 0; ir < N; ir++) {
        zm4x4_eq_spin_projection_zm4x4_32 (  sp_out[ir], sp_in[ir], a, b );
      }
      break;;
    default:
      fprintf(stderr, "[correlator_spin_projection] Error, projector P_{%d,%d} not implemented\n", i, k);
      return(1);
      break;;
  }  /* end of switch i, k */
  return(0);
}  /* end of correlator_spin_projection */

int correlator_spin_parity_projection (double _Complex ***sp_out, double _Complex ***sp_in, double c, unsigned N) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ir = 0; ir < N; ir++) {
    zm4x4_eq_spin_parity_projection_zm4x4 ( sp_out[ir], sp_in[ir], c);
  }
  return(0);
}  /* end of correlator_spin_parity_projection */

}  /* end of namespace cvc */