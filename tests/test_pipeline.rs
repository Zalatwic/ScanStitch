mod common;
use common::synthetic;
use tempfile::TempDir;

#[test]
fn test_full_pipeline_no_stitch() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp1 = synthetic::film_negative_image(
        200, 400, 10, 30,
        [12000, 7000, 3000],
        [5000, 4000, 3500],
    );
    let comp2 = synthetic::film_negative_image(
        200, 400, 10, 30,
        [12000, 7000, 3000],
        [6000, 4500, 3200],
    );

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp1, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp2, &path2).unwrap();

    let output_dir = tmp.path().join("output");

    let cli = scanstitch::cli::Cli {
        component1: path1,
        component2: path2,
        output_dir: output_dir.clone(),
        debug: true,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    assert!(!report.phases.is_empty());
    assert!(output_dir.join("output.tiff").exists());
}
