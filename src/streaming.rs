use ndarray::{s, ArrayView3};

/// Default number of rows per processing tile.
pub const DEFAULT_TILE_ROWS: usize = 512;

/// Compute tile row ranges for a given image height and tile size.
/// Returns Vec of (start_row, end_row) pairs where end_row is exclusive.
pub fn tile_ranges(height: usize, tile_rows: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < height {
        let end = (start + tile_rows).min(height);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// Extract a tile (row slice) from an Array3 as an ArrayView3.
/// The slice covers rows [start..end) across all columns and channels.
pub fn tile_view<T>(arr: &ndarray::Array3<T>, start: usize, end: usize) -> ArrayView3<'_, T> {
    arr.slice(s![start..end, .., ..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_ranges_exact() {
        let ranges = tile_ranges(1024, 512);
        assert_eq!(ranges, vec![(0, 512), (512, 1024)]);
    }

    #[test]
    fn test_tile_ranges_remainder() {
        let ranges = tile_ranges(1000, 512);
        assert_eq!(ranges, vec![(0, 512), (512, 1000)]);
    }

    #[test]
    fn test_tile_ranges_small() {
        let ranges = tile_ranges(100, 512);
        assert_eq!(ranges, vec![(0, 100)]);
    }

    #[test]
    fn test_tile_ranges_zero() {
        let ranges = tile_ranges(0, 512);
        assert!(ranges.is_empty());
    }
}
