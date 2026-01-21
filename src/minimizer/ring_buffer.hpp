#pragma once

#include <vector>
#include <cassert>
#include <stdexcept>

namespace cpu_mlp {

/**
 * @brief A fixed-capacity ring buffer (circular buffer).
 * @details Efficiently stores a sliding window of elements. 
 *          Pushing back when full overwrites the oldest element (logical index 0).
 *          Accessing index i corresponds to the (i)-th oldest element.
 */
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity = 0) 
        : _capacity(capacity), _head(0), _count(0) {
        if (_capacity > 0) {
            _data.resize(_capacity);
        }
    }

    void set_capacity(size_t capacity) {
        _capacity = capacity;
        _data.resize(capacity);
        _head = 0;
        _count = 0;
    }

    void push_back(const T& val) {
        if (_capacity == 0) return; // Or throw? For now silent no-op or assert.
        
        if (_count < _capacity) {
            // Not full: append at (_head + _count) % _capacity
            // Since _head is usually 0 initially, this fills sequentially.
            // But if we erased (not supported here) or wrapped, we must be careful.
            // For this simple implementation:
            size_t insert_idx = (_head + _count) % _capacity;
            _data[insert_idx] = val;
            _count++;
        } else {
            // Full: overwrite head, and move head forward
            _data[_head] = val;
            _head = (_head + 1) % _capacity;
        }
    }

    // Access by logical index: 0 is oldest, size()-1 is newest
    T& operator[](size_t i) {
        assert(i < _count);
        return _data[(_head + i) % _capacity];
    }

    const T& operator[](size_t i) const {
        assert(i < _count);
        return _data[(_head + i) % _capacity];
    }

    T& back() {
        assert(_count > 0);
        return (*this)[_count - 1];
    }

    const T& back() const {
        assert(_count > 0);
        return (*this)[_count - 1];
    }

    size_t size() const { return _count; }
    bool empty() const { return _count == 0; }
    bool full() const { return _count == _capacity; }
    
    void clear() {
        _count = 0;
        _head = 0;
    }
    
    // For convenience, to mimic vector::reserve (though we use resizing in constructor)
    void reserve(size_t n) {
        if (n > _capacity) {
            set_capacity(n); // Warning: this clears data in my simple implementation! 
                             // LBFGS usually sets size once.
        }
    }

private:
    std::vector<T> _data;
    size_t _capacity;
    size_t _head;  // Index of the oldest element
    size_t _count; // Current number of elements
};

} // namespace cpu_mlp
