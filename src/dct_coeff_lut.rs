//! Decode tables for the run-level `dct_coeff` codeword tables:
//! ISO/IEC 11172-2 Tables B.5c / B.5d / B.5e ([`crate::dct_coeff`])
//! and ISO/IEC 13818-2 Tables B-14 / B-15 ([`crate::mpeg2_dct_coeff`]).
//!
//! Every codeword in those tables (sign bit excluded) is a run of
//! leading zeros, a `1`, then a short tail: at most 4 bits after 11
//! leading zeros in the 16-bit rows, at most 7 bits after none. A
//! [`CoeffLut`] groups the rows by their number of leading zeros and
//! gives each group one slot for every value of its longest tail, so
//! a single lookup on the next 16 bits of the stream finds the row
//! whose codeword starts them. The walkers build one table per
//! (table, position) from their own rows, on first use, and fall back
//! to their longest-first walk only near the end of the buffer, where
//! fewer than 16 bits plus a sign bit remain.

/// The rows of one codeword table, indexed by the codeword that starts
/// a 16-bit window. `T` is the walker's own row type.
pub(crate) struct CoeffLut<T> {
    /// `groups[z]` is the offset into `slots` of the group of
    /// codewords with `z` leading zeros, and the width of the longest
    /// tail in that group. The group holds `1 << width` slots.
    groups: [(u16, u8); 16],
    /// The row each window selects, or `None` where no codeword
    /// starts the window.
    slots: Vec<Option<T>>,
}

impl<T: Copy> CoeffLut<T> {
    /// Build the table from `(code, bits, row)` triples, where `code`
    /// is the codeword MSB-first and right-justified in `bits` bits
    /// (`1..=16`). Where one codeword is a prefix of another, the
    /// longer row wins, as in a longest-first walk.
    pub(crate) fn new(rows: impl IntoIterator<Item = (u16, u8, T)>) -> Self {
        let mut rows: Vec<(u16, u8, T)> = rows.into_iter().collect();
        // (leading zeros, tail width) of a codeword.
        let shape = |code: u16, bits: u8| {
            let significant = 16 - code.leading_zeros();
            debug_assert!(code != 0 && significant <= u32::from(bits) && bits <= 16);
            let zeros = u32::from(bits) - significant;
            (zeros as usize, significant - 1)
        };
        let mut groups = [(0u16, 0u8); 16];
        for &(code, bits, _) in &rows {
            let (zeros, tail) = shape(code, bits);
            groups[zeros].1 = groups[zeros].1.max(tail as u8);
        }
        let mut len = 0u16;
        for group in &mut groups {
            group.0 = len;
            len += 1 << group.1;
        }
        let mut slots = vec![None; usize::from(len)];
        // Shortest first, so a longer row overwrites the slots of a
        // shorter row that is a prefix of it.
        rows.sort_by_key(|&(_, bits, _)| bits);
        for (code, bits, row) in rows {
            let (zeros, tail) = shape(code, bits);
            let (offset, width) = groups[zeros];
            let free = u32::from(width) - tail;
            let tail_value = u32::from(code) & ((1 << tail) - 1);
            let start = usize::from(offset) + (tail_value << free) as usize;
            slots[start..start + (1 << free)].fill(Some(row));
        }
        Self { groups, slots }
    }

    /// The row whose codeword starts `window` (the next 16 bits,
    /// MSB-first), if any.
    #[inline]
    pub(crate) fn lookup(&self, window: u16) -> Option<T> {
        let zeros = window.leading_zeros();
        if zeros >= 16 {
            return None;
        }
        let (offset, width) = self.groups[zeros as usize];
        // The bits after the leading `1`, top-aligned in 16 bits, then
        // cut to the group's tail width.
        let tail = ((u32::from(window) << (zeros + 1)) & 0xffff) >> (16 - u32::from(width));
        self.slots[usize::from(offset) + tail as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longer_row_wins_over_its_prefix() {
        // `01` and `0101`: the longer row owns the windows it starts.
        let lut = CoeffLut::new([(0b01, 2, 'a'), (0b0101, 4, 'b')]);
        assert_eq!(lut.lookup(0b0101_0000_0000_0000), Some('b'));
        assert_eq!(lut.lookup(0b0100_0000_0000_0000), Some('a'));
        assert_eq!(lut.lookup(0b0111_1111_1111_1111), Some('a'));
        assert_eq!(lut.lookup(0b1000_0000_0000_0000), None);
        assert_eq!(lut.lookup(0), None);
    }

    #[test]
    fn sixteen_bit_codeword() {
        let lut = CoeffLut::new([(0b0000_0000_0001_1111, 16, 1u8), (0b1, 1, 2)]);
        assert_eq!(lut.lookup(0b0000_0000_0001_1111), Some(1));
        assert_eq!(lut.lookup(0b0000_0000_0001_1110), None);
        assert_eq!(lut.lookup(0b1111_1111_1111_1111), Some(2));
    }
}
