/// Minimum transmittance to avoid log(0) in density calculations.
pub const EPSILON_T: f64 = 1e-6;

/// Maximum value for 14-bit data.
pub const MAX_14BIT: f64 = 16383.0;

/// Maximum value for 16-bit data.
pub const MAX_16BIT: f64 = 65535.0;

/// D50 illuminant white point in CIE XYZ.
pub const D50_WHITE: [f64; 3] = [0.9642, 1.0000, 0.8251];

/// ProPhoto RGB to CIE XYZ (D50) matrix.
pub const PROPHOTO_TO_XYZ_D50: [[f64; 3]; 3] = [
    [0.7977, 0.1352, 0.0313],
    [0.2880, 0.7119, 0.0001],
    [0.0000, 0.0000, 0.8251],
];

/// CIE XYZ (D50) to ProPhoto RGB matrix (inverse of above).
pub const XYZ_D50_TO_PROPHOTO: [[f64; 3]; 3] = [
    [1.345886547039, -0.255603120044, -0.051024952867],
    [-0.544480019030, 1.508096219375, 0.020471960943],
    [0.000000000000, 0.000000000000, 1.211974306145],
];

/// Bradford chromatic adaptation: XYZ to LMS.
pub const BRADFORD_XYZ_TO_LMS: [[f64; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Bradford chromatic adaptation: LMS to XYZ.
pub const BRADFORD_LMS_TO_XYZ: [[f64; 3]; 3] = [
    [0.9870, -0.1471, 0.1600],
    [0.4323, 0.5184, 0.0493],
    [-0.0085, 0.0400, 0.9685],
];
