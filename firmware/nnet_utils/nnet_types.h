#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <assert.h>
#include <cstddef>
#include <cstdio>

#ifdef __SYNTHESIS__
#define SYN_PRAGMA(PRAG) _Pragma(#PRAG)
#else
#define SYN_PRAGMA(PRAG)
#endif


namespace nnet {

template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;

    T data[N];

protected:
  /// Pragma setter (hack until we support pragma on types)
  /// Note: must be used on all functions if possible
  INLINE void pragma() const {
    SYN_PRAGMA(HLS DISAGGREGATE variable=this)
  }


public:
    INLINE array() {
       pragma();
       SYN_PRAGMA(HLS ARRAY_PARTITION variable=this->data complete)
    }

    INLINE T &operator[](size_t pos) {
    	pragma();
    	return data[pos];
    }

    INLINE const T &operator[](size_t pos) const {
    	pragma();
    	return data[pos];
    }

    INLINE array &operator=(const array &other) {
    	pragma();
    	other.pragma();

        assert(N == other.size && "Array sizes must match.");

        for (unsigned i = 0; i < N; i++) {
        	SYN_PRAGMA(HLS UNROLL)
            data[i] = other[i];
        }
        return *this;
    }
};

// Generic lookup-table implementation, for use in approximations of math functions
template <typename T, unsigned N, T (*func)(T)> class lookup_table {
  public:
    lookup_table(T from, T to) : range_start(from), range_end(to), base_div(ap_uint<16>(N) / T(to - from)) {
        T step = (range_end - range_start) / ap_uint<16>(N);
        for (size_t i = 0; i < N; i++) {
            T num = range_start + ap_uint<16>(i) * step;
            T sample = func(num);
            samples[i] = sample;
        }
    }

    T operator()(T n) const {
        int index = (n - range_start) * base_div;
        if (index < 0)
            index = 0;
        else if (index > N - 1)
            index = N - 1;
        return samples[index];
    }

  private:
    T samples[N];
    const T range_start, range_end;
    ap_fixed<20, 16> base_div;
};

} // namespace nnet

#endif
