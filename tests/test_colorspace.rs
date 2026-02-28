use approx::assert_relative_eq;
use nalgebra::Vector3;

#[test]
fn test_bradford_adaptation_d65_to_d50() {
    let source_white = [0.9505, 1.0000, 1.0890];
    let cat = scanstitch::colorspace::bradford_cat(&source_white);
    let xyz_d65 = Vector3::new(0.9505, 1.0, 1.0890);
    let xyz_d50 = cat * xyz_d65;
    assert_relative_eq!(xyz_d50[0], 0.9642, epsilon = 0.01);
    assert_relative_eq!(xyz_d50[1], 1.0, epsilon = 0.01);
    assert_relative_eq!(xyz_d50[2], 0.8251, epsilon = 0.01);
}

#[test]
fn test_prophoto_roundtrip() {
    let xyz_to_pro = scanstitch::colorspace::xyz_d50_to_prophoto_matrix();
    let pro_to_xyz = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    let xyz = Vector3::new(0.5, 0.4, 0.3);
    let pro = xyz_to_pro * xyz;
    let xyz2 = pro_to_xyz * pro;
    for i in 0..3 {
        assert_relative_eq!(xyz[i], xyz2[i], epsilon = 1e-4);
    }
}

#[test]
fn test_output_clamped_0_1() {
    let mut img = ndarray::Array3::<f64>::zeros((10, 10, 3));
    for y in 0..10 {
        for x in 0..10 {
            img[[y, x, 0]] = 0.8;
            img[[y, x, 1]] = 0.5;
            img[[y, x, 2]] = 0.3;
        }
    }
    let result = scanstitch::colorspace::map_to_prophoto_d50(&img);
    for y in 0..10 {
        for x in 0..10 {
            for c in 0..3 {
                let v = result[[y, x, c]];
                assert!(
                    v >= 0.0 && v <= 1.0,
                    "out of range: {} at ({},{},{})",
                    v,
                    y,
                    x,
                    c
                );
            }
        }
    }
}
