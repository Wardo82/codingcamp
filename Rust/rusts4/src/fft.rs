/*
 * Candle has no FFT op, so we only get the nofft=True path natively.
 * For the tutorial's sequence lengths (tens to low hundreds), O(L²) is completely fine and lets you validate everything end-to-end.
 * I'll flag the FFT path as a later optimization rather than solve it now, since it's a bigger piece of work
 * (a CustomOp around rustfft, with a hand-written backward pass) and it doesn't change correctness, only speed at large L.
*/
