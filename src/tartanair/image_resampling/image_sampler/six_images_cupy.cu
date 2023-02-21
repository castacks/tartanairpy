
// #include <math_constants.h>

extern "C" __global__ void cu_sample_coor(
    int n_points,
    const float3* xyz, const float2* offsets, 
    float2* output, float2* out_offsets) {
    // Prepare the index.
    const int x_idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_stride = blockDim.x * gridDim.x;

    // Constants.
    const float one_fourth_pi   = 0.785398163F; // Defined as CUDART_PIO4_F in <math_constants.h>;
    const float half_pi         = 1.570796327F; // Defined as CUDART_PIO2_F in <math_constants.h>;
    const float three_fourth_pi = one_fourth_pi + half_pi;

    // Dimensionless image size.
    const float dls = 2.f;
    const float dls_half = dls / 2.f;

    // Loop.
    for ( int i = x_idx; i < n_points; i += x_stride ) {
        const float x = xyz[i].x;
        const float y = xyz[i].y;
        const float z = xyz[i].z;

        const float a_y     = atan2( x, y ); // Angle w.r.t. y+ axis projected to the x-y plane.
        const float a_z     = atan2( z, y ); // Angle w.r.t. y+ axis projected to the y-z plane.
        const float azimuth = atan2( z, x ); // Angle w.r.t. x+ axis projected to the z-x plane.

        out_offsets[i].x = 0.f;
        out_offsets[i].y = 0.f;

        if ( -one_fourth_pi < a_y && a_y < one_fourth_pi && \
             -one_fourth_pi < a_z && a_z < one_fourth_pi ) {
            // Bottom.
            output[i].x = min( max( ( dls_half + x/y ) / dls, 0.f ), 1.f );
            output[i].y = min( max( ( dls_half - z/y ) / dls, 0.f ), 1.f );
            out_offsets[i].x = offsets[2].x;
            out_offsets[i].y = offsets[2].y;
        } else if ( ( three_fourth_pi < a_y || a_y < -three_fourth_pi ) && \
                    ( three_fourth_pi < a_z || a_z < -three_fourth_pi ) ) {
            // Top.
            output[i].x = min( max( ( dls_half - x/y ) / dls, 0.f ), 1.f );
            output[i].y = min( max( ( dls_half - z/y ) / dls, 0.f ), 1.f );
            out_offsets[i].x = offsets[4].x;
            out_offsets[i].y = offsets[4].y;
        } else if ( one_fourth_pi <= azimuth && azimuth < three_fourth_pi ) {
            // Front.
            output[i].x = min( max( ( dls_half + x/z ) / dls, 0.f ), 1.f );
            output[i].y = min( max( ( dls_half + y/z ) / dls, 0.f ), 1.f );
            out_offsets[i].x = offsets[0].x;
            out_offsets[i].y = offsets[0].y;
        } else if ( -one_fourth_pi <= azimuth && azimuth < one_fourth_pi ) {
            // Right.
            output[i].x = min( max( ( dls_half - z/x ) / dls, 0.f ), 1.f );
            output[i].y = min( max( ( dls_half + y/x ) / dls, 0.f ), 1.f );
            out_offsets[i].x = offsets[1].x;
            out_offsets[i].y = offsets[1].y;
        } else if ( -three_fourth_pi <= azimuth && azimuth < -one_fourth_pi ) {
            // Back.
            output[i].x = min( max( ( dls_half + x/z ) / dls, 0.f ), 1.f );
            output[i].y = min( max( ( dls_half - y/z ) / dls, 0.f ), 1.f );
            out_offsets[i].x = offsets[5].x;
            out_offsets[i].y = offsets[5].y;
        } else if ( three_fourth_pi <= azimuth || azimuth < -three_fourth_pi ) {
            // Left.
            output[i].x = min( max( ( dls_half - z/x ) / dls, 0.f ), 1.f );
            output[i].y = min( max( ( dls_half - y/x ) / dls, 0.f ), 1.f );
            out_offsets[i].x = offsets[3].x;
            out_offsets[i].y = offsets[3].y;
        }
    }
}
