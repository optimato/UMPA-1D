#include <iostream>
#include <vector>
#include <cmath>
#include <exception>
#include "Model.h"
#include "Optim.h"
#include "Utils.h"

using namespace std;

template <class T> CostArgs<T>::CostArgs() {};

template <class T> CostArgs<T>::CostArgs(int i, int j)
{
    this->ij[0] = i;
    this->ij[1] = j;
};

template <class T> CostArgs<T>::~CostArgs() {};

template <class T> CostArgsNoDF<T>::CostArgsNoDF() {};

template <class T> CostArgsNoDF<T>::CostArgsNoDF(int i, int j) : CostArgs<T>(i, j), t(0.) {};

template <class T> CostArgsNoDF<T>::~CostArgsNoDF() {};

template <class T> CostArgsDF<T>::CostArgsDF() {};

template <class T> CostArgsDF<T>::CostArgsDF(int i, int j) : CostArgs<T>(i, j), t(0.), v(0.) {};

template <class T> CostArgsDF<T>::~CostArgsDF() {};

template <class T> CostArgsDFKernel<T>::CostArgsDFKernel() {};

template <class T> CostArgsDFKernel<T>::CostArgsDFKernel(int i, int j, T a, T b, T c) :
    CostArgs<T>(i, j),
    t(0.),
    a(a),
    b(b),
    c(c)
{
    T norm = 0;

    /* Generate kernel*/
    for (int k = 0; k < 2 * KERNEL_WINDOW_SIZE + 1; k++)     // kernel rows
    {
        for (int l = 0; l < 2 * KERNEL_WINDOW_SIZE + 1; l++) // kernel columns
        {
            kernel[k * (2 * KERNEL_WINDOW_SIZE + 1) + l] = gaussian_kernel<T>(k - KERNEL_WINDOW_SIZE, l - KERNEL_WINDOW_SIZE, a, b, c);
            norm += kernel[k * (2 * KERNEL_WINDOW_SIZE + 1) + l];
        }
    }

    /* Normalise */
    for (int k = 0; k < 2 * KERNEL_WINDOW_SIZE + 1; k++)     // kernel rows
    {
        for (int l = 0; l < 2 * KERNEL_WINDOW_SIZE + 1; l++) // kernel columns
        {
            kernel[k * (2 * KERNEL_WINDOW_SIZE + 1) + l] /= norm;
        }
    }
};

template <class T> CostArgsDFKernel<T>& CostArgsDFKernel<T>::operator=(const CostArgsDFKernel<T>& rhs)
{
    if (this != &rhs)
    {
        t = rhs.t;
        a = rhs.a;
        b = rhs.b;
        c = rhs.c;
        this->ij[0] = rhs.ij[0];
        this->ij[1] = rhs.ij[1];
        for (int i = 0; i < (2 * KERNEL_WINDOW_SIZE + 1) * (2 * KERNEL_WINDOW_SIZE + 1); i++)
        {
            kernel[i] = rhs.kernel[i];
        }
    }
    return *this;
};


template <class T> CostArgsDFKernel<T>::~CostArgsDFKernel() {};

namespace models {

    /*
     * Model Base
     */

     // Nullary constructor
    template <class T> ModelBase<T>::ModelBase() {}

    /*
     * Constructor
     * Arguments:
     *  - Na: number of frames
     *  - dim: frame dimensions
     *  - sams, refs, masks: vectors containing frame pointers
     *  - Nw: window width
     *  - win: window pointer
     *  - max_shift: default maximum shift
     */
    template <class T>
    ModelBase<T>::ModelBase(int Na, vector<int*>& dim, vector<T*>& sams, vector<T*>& refs, vector<T*>& masks, vector<int*>& pos, int Nw, T* win, int max_shift, int padding) :
        Na(Na),
        dim{ dim },
        Nw(Nw),
        max_shift(max_shift),
        sam{ sams },
        ref{ refs },
        mask{ masks },
        win(win),
        pos{ pos },
        padding(padding),
        reference_shift(1)
    {}

    // Destructor
    template <class T> ModelBase<T>::~ModelBase() {}

    template <class T>
    T ModelBase<T>::test()
    {
        cout << this->sam[0] << " address of first element." << endl;
        cout << this->sam[0][0] << " first element." << endl;
        cout << this->dim[0][0] << ", " << this->dim[0][1] << " dimensions." << endl;
        return (T)sam.size();
    }

    /*
    Set or reset the window pointer and dimensions
    */
    template <class T>
    void ModelBase<T>::set_window(T* new_win, int new_Nw)
    {
        if (new_Nw < 0) throw runtime_error("Nw must be non-negative.");
        this->win = new_win;
        this->Nw = new_Nw;
        return;
    }

    /*
     * Coverage map
     * Input variables:
     *  - out: pointer for output value of the coverage (sum of valid frames - eventually weighted with masks)
     *  - i, j: image coordinates
    */
    template <class T>
    error_status ModelBase<T>::coverage(T* out, int i, int j)
    {
        int kk;
        T c = 0;
        error_status s = { 0 };

        for (kk = 0; kk < this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if ((i - pos_i - this->padding) < 0) continue;
            if ((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if ((j - pos_j - this->padding) < 0) continue;
            if ((j - pos_j + this->padding) > this->dim[kk][1]) continue;
            c += 1.;
        }
        *out = c;
        s.ok = 1;
        return s;
    }

    template <class T>
    error_status ModelBase<T>::min(int i, int j, T* values, minimizer_debug<T>* db)
    {
        T uv[2] = { 0. };
        return min(i, j, values, uv, db);
    }


    /*
     * UMPA model without dark field and without masking
     * Input variables:
     *  - out: pointer for output value of the cost function
     *  - ij[2]: displacement between reference and sample arrays  //fdm: isn't this shift_ij?
     *  - args[0], args[1]: input parameter: pixel position in the frame //fdm: isn't it args->ij?
     *  - args[2]: output value: sample transmission
    */
    template <class T>
    error_status ModelNoDF<T>::cost(T* out, int* shift_ij, CostArgsNoDF<T>* args)
    {
        int i, j, iu, ju, ia, ib, ja, jb;
        int ii, jj, kk;
        T t1 = 0.;
        T t3 = 0.;
        T t5 = 0.;
        error_status s = { 0 };

        if (shift_ij[0] <= -this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 0;
            s.positive = 0;
            return s;
        }
        if (shift_ij[0] >= this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 0;
            s.positive = 0;
            return s;
        }
        if (shift_ij[1] <= -this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 1;
            s.positive = 0;
            return s;
        }
        if (shift_ij[1] >= this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 1;
            s.positive = 1;
            return s;
        }

        // Pixel coordinate
        i = (int)round(args->ij[0]);
        j = (int)round(args->ij[1]);

        // Offset coordinate
        if (this->reference_shift)
        {
            iu = i - shift_ij[0];
            ju = j - shift_ij[1];
            ia = i;
            ja = j;
            ib = iu;
            jb = ju;
        }
        else
        {
            iu = i + shift_ij[0];
            ju = j + shift_ij[1];
            ia = iu;
            ja = ju;
            ib = i;
            jb = j;
        }

        for (kk = 0; kk < this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if ((i - pos_i - this->padding) < 0) continue;
            if ((i - pos_i + this->padding) > this->dim[kk][0]) continue;  // fdm: not dim[kk][0]-1?
            if ((j - pos_j - this->padding) < 0) continue;
            if ((j - pos_j + this->padding) > this->dim[kk][1]) continue;
            for (ii = 0; ii < 2 * this->Nw + 1; ii++)
            {
                for (jj = 0; jj < 2 * this->Nw + 1; jj++)
                {
                    t1 += this->win[ii * (2 * this->Nw + 1) + jj] * this->sam[kk][(ii - this->Nw + ib - pos_i) * this->dim[kk][1] + (jj - this->Nw + jb - pos_j)] * this->sam[kk][(ii - this->Nw + ib - pos_i) * this->dim[kk][1] + (jj - this->Nw + jb - pos_j)];
                    t3 += this->win[ii * (2 * this->Nw + 1) + jj] * this->ref[kk][(ii - this->Nw + ia - pos_i) * this->dim[kk][1] + (jj - this->Nw + ja - pos_j)] * this->ref[kk][(ii - this->Nw + ia - pos_i) * this->dim[kk][1] + (jj - this->Nw + ja - pos_j)];
                    t5 += this->win[ii * (2 * this->Nw + 1) + jj] * this->ref[kk][(ii - this->Nw + ia - pos_i) * this->dim[kk][1] + (jj - this->Nw + ja - pos_j)] * this->sam[kk][(ii - this->Nw + ib - pos_i) * this->dim[kk][1] + (jj - this->Nw + jb - pos_j)];
                }
            }
        }

        // Set transmission function
        args->t = t5 / t3;

        // Set cost function value
        *out = t1 - t5 * args->t;    

        s.ok = 1;
        return s;
    }

    /*
     * Cost function for outside interface.
     * Input variables:
     *  - int i, int j: pixel coordinates in sample frame
     *  - shift_i, shift_j: displacement between reference and sample frames
     * Output arguments:
     *  - values[0]: cost function value
     *  - values[1]: transmission
    */
    template <class T>
    error_status ModelNoDF<T>::cost_interface(int i, int j, int shift_i, int shift_j, T* values)
    {
        int shift_ij[2] = { shift_i, shift_j };
        CostArgsNoDF<T> args = CostArgsNoDF<T>(i, j);
        error_status s = this->cost(&values[0], shift_ij, &args);
        values[1] = args.t;
        return s;
    }

    /*
     * UMPA optimisation on a single pixel (no dark field, no masking)
     *
     *    Arguments:
     *  i, j: input - pixel position
     *  values: {D, t, sx, sx,} (cost function, sample transmission, dpcx, dpcy)
     */
    template <class T>
    error_status ModelNoDF<T>::min(int i, int j, T* values, T uv[2], minimizer_debug<T>* db)
    {
        T D;
        error_status s;
        CostArgsNoDF<T> args = CostArgsNoDF<T>(i, j);

        s = discrete_1d_minimizer<T, ModelNoDF, CostArgsNoDF<T>, &ModelNoDF::cost>(&D, &uv[0], this, &args, db);
        //s = discrete_2d_minimizer<T,int(ModelNoDF::cost)(int,int) >(D, &uv[0], [=](T* out, int* ij, T* args) -> error_status { return this->cost(out, ij, args); }, N, args, db);
        values[0] = D;
        values[1] = args.t;
        values[2] = uv[1];
        values[3] = uv[0];
        return s;
    }

    /*******************************************
     *
     *  MODEL WITH DARK FIELD
     *
     *******************************************/


    template <class T>
    ModelDF<T>::ModelDF(int Na, vector<int*>& dim, vector<T*>& sams, vector<T*>& refs, vector<T*>& masks, vector<int*>& pos, int Nw, T* win, int max_shift, int padding) :
        ModelBase<T>(Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding)
    {
        T denom = 0;
        this->Im = 0.;
        for (int kk = 0; kk < this->Na; kk++)
        {
            for (int ii = 0; ii < this->dim[kk][0]; ii++)
            {
                for (int jj = 0; jj < this->dim[kk][1]; jj++)
                {
                    this->Im += this->ref[kk][ii * this->dim[kk][1] + jj];
                    denom += 1.;
                }
            }
        }
        this->Im /= denom;
    }

    /*
     * UMPA model WITH dark field and without masking
     * Input variables:
     *  - out: pointer for output value of the cost function
     *  - shift_ij[2]: displacement between reference and sample arrays
     *  - args[0], args[1]: input parameter: pixel position in the frame
     *  - args[2]: output value: sample transmission
     *  - args[3]: output value: dark field
    */
    template <class T>
    error_status ModelDF<T>::cost(T* out, int* shift_ij, CostArgsDF<T>* args)
    {
        int i, j, iu, ju, ia, ib, ja, jb;
        int ii, jj, kk;
        T t1 = 0.;
        T t2 = 0.;
        T t3 = 0.;
        T t4 = 0.;
        T t5 = 0.;
        T t6 = 0.;
        T t4_term = 0.;   // variable for intermediate summation step
        T t6_term = 0.;   // variable for intermediate summation step
        T ref_mean = 0.;  // mean of ref intensity over window
        //T ref_denom = 0.;  // number of pixels in the mean calculation
        T beta, K;
        error_status s = { 0 };
        int Nw = this->Nw;

        if (shift_ij[0] <= -this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 0;
            s.positive = 0;
            return s;
        }
        if (shift_ij[0] >= this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 0;
            s.positive = 0;
            return s;
        }
        if (shift_ij[1] <= -this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 1;
            s.positive = 0;
            return s;
        }
        if (shift_ij[1] >= this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 1;
            s.positive = 1;
            return s;
        }

        // Pixel coordinate
        i = (int)round(args->ij[0]);
        j = (int)round(args->ij[1]);

        // Offset coordinate
        if (this->reference_shift)
        {
            iu = i - shift_ij[0];
            ju = j - shift_ij[1];
            ia = i;
            ja = j;
            ib = iu;
            jb = ju;
        }
        else
        {
            iu = i + shift_ij[0];
            ju = j + shift_ij[1];
            ia = iu;
            ja = ju;
            ib = i;
            jb = j;
        }

        for (kk = 0; kk < this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if ((i - pos_i - this->padding) < 0) continue;
            if ((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if ((j - pos_j - this->padding) < 0) continue;
            if ((j - pos_j + this->padding) > this->dim[kk][1]) continue;

            // Change due to per-frame means: calculate the per-frame mean
            // before the cost function terms!
            ref_mean = 0.;
            for (ii = 0; ii < 2 * Nw + 1; ii++)
            {
                for (jj = 0; jj < 2 * Nw + 1; jj++)
                {
                    ref_mean += this->ref[kk][(ii - Nw + ia - pos_i) * this->dim[kk][1] + (jj - Nw + ja - pos_j)];
                }
            }
            ref_mean /= (2 * Nw + 1) * (2 * Nw + 1);

            // additional change: The summation of the t4 and t6 terms
            // must be split: sum the terms over the window in the current
            // frame, then multiply with the mean in the frame's window.
            // i.e. t4 changed from: <I0> * sum_k,i,j(I_kij) to: sum_k <I0>_k sum_i,j I_k,i,j
            t4_term = 0.;
            t6_term = 0.;
            for (ii = 0; ii < 2 * Nw + 1; ii++)
            {
                for (jj = 0; jj < 2 * Nw + 1; jj++)
                {
                    t1 += this->win[ii * (2 * Nw + 1) + jj] * this->sam[kk][(ii - Nw + ib - pos_i) * this->dim[kk][1] + (jj - Nw + jb - pos_j)] * this->sam[kk][(ii - Nw + ib - pos_i) * this->dim[kk][1] + (jj - Nw + jb - pos_j)];
                    t3 += this->win[ii * (2 * Nw + 1) + jj] * this->ref[kk][(ii - Nw + ia - pos_i) * this->dim[kk][1] + (jj - Nw + ja - pos_j)] * this->ref[kk][(ii - Nw + ia - pos_i) * this->dim[kk][1] + (jj - Nw + ja - pos_j)];
                    t4_term += this->win[ii * (2 * Nw + 1) + jj] * this->sam[kk][(ii - Nw + ib - pos_i) * this->dim[kk][1] + (jj - Nw + jb - pos_j)];
                    t5 += this->win[ii * (2 * Nw + 1) + jj] * this->ref[kk][(ii - Nw + ia - pos_i) * this->dim[kk][1] + (jj - Nw + ja - pos_j)] * this->sam[kk][(ii - Nw + ib - pos_i) * this->dim[kk][1] + (jj - Nw + jb - pos_j)];
                    t6_term += this->win[ii * (2 * Nw + 1) + jj] * this->ref[kk][(ii - Nw + ia - pos_i) * this->dim[kk][1] + (jj - Nw + ja - pos_j)];
                }
            }
            t2 += ref_mean * ref_mean;
            t4 += ref_mean * t4_term;
            t6 += ref_mean * t6_term;
        }

        K = (t2 * t5 - t4 * t6) / (t2 * t3 - t6 * t6);
        beta = (t3 * t4 - t5 * t6) / (t2 * t3 - t6 * t6);

        // Transmission function and dark field
        args->t = beta + K;
        args->v = K / args->t;


        // Set cost function value
        *out = t1 + beta * beta * t2 + K * K * t3 - 2 * beta * t4 - 2 * K * t5 + 2 * beta * K * t6;

        s.ok = 1;
        return s;
    }

    /*
     * Cost function for outside interface.
     * Input variables:
     *  - int i, int j: pixel coordinates in sample frame
     *  - shift_i, shift_j: displacement between reference and sample frames
     * Output arguments:
     *  - values[0]: cost function value
     *  - values[1]: transmission
     *  - values[2]: dark-field
    */
    template <class T>
    error_status ModelDF<T>::cost_interface(int i, int j, int shift_i, int shift_j, T* values)
    {
        int shift_ij[2] = { shift_i, shift_j };
        CostArgsDF<T> args = CostArgsDF<T>(i, j);
        error_status s = this->cost(&values[0], shift_ij, &args);
        values[1] = args.t;
        values[2] = args.v;
        return s;
    }


    /*
     * UMPA optimisation on a single pixel (no dark field, no masking)
     *
     *    Arguments:
     *  i, j: input - pixel position
     *  values: {D, t, sx, sx, v} (cost function, sample transmission, dpcx, dpcy, dark-field)
     */
    template <class T>
    error_status ModelDF<T>::min(int i, int j, T* values, T uv[2], minimizer_debug<T>* db)
    {
        T D;
        error_status s;

        CostArgsDF<T> args = CostArgsDF<T>(i, j);

        s = discrete_1d_minimizer<T, ModelDF, CostArgsDF<T>, &ModelDF::cost>(&D, &uv[0], this, &args, db);

        values[0] = D;
        values[1] = args.t;
        values[2] = uv[1];
        values[3] = uv[0];
        values[4] = args.v;
        return s;
    }


    /*******************************************
     *
     *  MODEL WITH 3-PARAMETER KERNEL DARK FIELD
     *
     *******************************************/

    template <class T>
    ModelDFKernel<T>::ModelDFKernel(int Na, vector<int*>& dim, vector<T*>& sams, vector<T*>& refs, vector<T*>& masks, vector<int*>& pos, int Nw, T* win, int max_shift, int padding) :
        ModelBase<T>(Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding), Nk(KERNEL_WINDOW_SIZE) { }
    /*
     * UMPA model without dark field and without masking
     * Input variables:
     *  - out: pointer for output value of the cost function
     *  - ij[2]: displacement between reference and sample arrays
     *  - args[0], args[1]: input parameter: pixel position in the frame
     *  - args[2]: output value: sample transmission
    */
    template <class T>
    error_status ModelDFKernel<T>::cost(T* out, int* shift_ij, CostArgsDFKernel<T>* args)
    {
        int i, j, iu, ju, ia, ib, ja, jb;
        int ii, jj, kk;
        T t1 = 0.;
        T t3 = 0.;
        T t5 = 0.;
        T blurred_ref;
        error_status s = { 0 };

        if (shift_ij[0] <= -this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 0;
            s.positive = 0;
            return s;
        }
        if (shift_ij[0] >= this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 0;
            s.positive = 0;
            return s;
        }
        if (shift_ij[1] <= -this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 1;
            s.positive = 0;
            return s;
        }
        if (shift_ij[1] >= this->max_shift)
        {
            s.bound_error = 1;
            s.dimension = 1;
            s.positive = 1;
            return s;
        }

        // Pixel coordinate
        i = (int)round(args->ij[0]);
        j = (int)round(args->ij[1]);

        // Offset coordinate
        if (this->reference_shift)
        {
            iu = i - shift_ij[0];
            ju = j - shift_ij[1];
            ia = i;
            ja = j;
            ib = iu;
            jb = ju;
        }
        else
        {
            iu = i + shift_ij[0];
            ju = j + shift_ij[1];
            ia = iu;
            ja = ju;
            ib = i;
            jb = j;
        }

        for (kk = 0; kk < this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if ((i - pos_i - this->padding) < 0) continue;
            if ((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if ((j - pos_j - this->padding) < 0) continue;
            if ((j - pos_j + this->padding) > this->dim[kk][1]) continue;

            for (ii = 0; ii < 2 * this->Nw + 1; ii++)
            {
                for (jj = 0; jj < 2 * this->Nw + 1; jj++)
                {
                    blurred_ref = convolve<T>(this->ref[kk], ii - this->Nw + ia - pos_i, jj - this->Nw + ja - pos_j, this->dim[kk], args->kernel, this->Nk);
                    t1 += this->win[ii * (2 * this->Nw + 1) + jj] * this->sam[kk][(ii - this->Nw + ib - pos_i) * this->dim[kk][1] + (jj - this->Nw + jb - pos_j)] * this->sam[kk][(ii - this->Nw + ib - pos_i) * this->dim[kk][1] + (jj - this->Nw + jb - pos_j)];
                    t3 += this->win[ii * (2 * this->Nw + 1) + jj] * blurred_ref * blurred_ref;
                    t5 += this->win[ii * (2 * this->Nw + 1) + jj] * blurred_ref * this->sam[kk][(ii - this->Nw + ib - pos_i) * this->dim[kk][1] + (jj - this->Nw + jb - pos_j)];
                }
            }
        }
        // Set transmission function
        args->t = t5 / t3;

        // Set cost function value
        *out = t1 - t5 * args->t;

        s.ok = 1;
        return s;
    }

    /*
     * Cost function for outside interface.
     * Input variables:
     *  - int i, int j: pixel coordinates in sample frame
     *  - shift_i, shift_j: displacement between reference and sample frames
     *  - values[2,3,4]: a,b,c for kernel
     * Output arguments:
     *  - values[0]: cost function value
     *  - values[1]: transmission
    */
    template <class T>
    error_status ModelDFKernel<T>::cost_interface(int i, int j, int shift_i, int shift_j, T* values)
    {
        int shift_ij[2] = { shift_i, shift_j };
        CostArgsDFKernel<T> args = CostArgsDFKernel<T>(i, j, values[2], values[3], values[4]);
        error_status s = this->cost(&values[0], shift_ij, &args);
        values[1] = args.t;
        return s;
    }

    /*
     * UMPA optimisation on a single pixel
     *
     *    Arguments:
     *  i, j: input - pixel position
     *  values: {D, t, sx, sx, a, b, c} (cost function, sample transmission, dpcx, dpcy)
     */
    template <class T>
    error_status ModelDFKernel<T>::min(int i, int j, T* values, T uv[2], minimizer_debug<T>* db)
    {
        T D;
        error_status s;
        CostArgsDFKernel<T> args = CostArgsDFKernel<T>(i, j, values[4], values[5], values[6]);

        s = discrete_1d_minimizer<T, ModelDFKernel, CostArgsDFKernel<T>, &ModelDFKernel::cost>(&D, &uv[0], this, &args, db);
        values[0] = D;
        values[1] = args.t;
        values[2] = uv[1];
        values[3] = uv[0];
        return s;
    }
}