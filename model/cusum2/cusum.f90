module cusum

contains

    function log_likelihood(x, lmbda)
        !! Poisson log-likelihood up to a constant
        implicit none
        real(kind=8), intent(in) :: x, lmbda
        real(kind=8) :: log_likelihood

        log_likelihood = -lmbda + x*log(lmbda)
        
    end function log_likelihood

    function log_likelihood_ratio(x, in_control, out_control)
      !! Difference between the log-likelihoods of the in-control and out-of-control rate models  
      implicit none
      real(kind=8), intent(in) :: x, in_control, out_control
      real(kind=8) :: numerator, denominator, log_likelihood_ratio
      numerator = log_likelihood(x, out_control)
      denominator = log_likelihood(x, in_control)

      log_likelihood_ratio = numerator - denominator
      
    end function log_likelihood_ratio

    subroutine control_chart(time_series, expectation, alpha, S)
      implicit none
      real(kind=8), intent(in) :: alpha
      real(kind=8), dimension(:), intent(in) :: time_series, expectation
      !f2py depend(time_series) S
      real(kind=8), intent(out), dimension(size(time_series)) :: S
      integer :: i, series_len

      series_len = size(time_series)

      S(1) = log_likelihood_ratio(time_series(1), expectation(1), expectation(1)*alpha)
      do i=2,series_len
         S(i) = max(0d0, log_likelihood_ratio(time_series(i), expectation(i), expectation(i)*alpha))
      end do
      
    end subroutine control_chart

    function flag(chart, h)
      implicit none
      real(kind=8), intent(in), dimension(:) :: chart
      real(kind=8), intent(in) :: h
      logical :: flag

      flag = maxval(chart) > h

    end function flag

    subroutine false_positive(simulated_series, expectation, alpha, h_values, fp_rate)
      implicit none
      real(kind=8), intent(in) :: alpha
      real(kind=8), intent(in), dimension(:) :: expectation, h_values
      real(kind=8), intent(in), dimension(:, :) :: simulated_series
      !f2py depend(h_values) fp_rate
      real(kind=8), intent(out), dimension(size(h_values)) :: fp_rate
      integer :: i, j, h_values_len, series_len, num_series
      real(kind=8) :: h
      real(kind=8), dimension(:), allocatable :: chart, ones_list, zeros_list
      logical, dimension(:), allocatable :: flag_list

      h_values_len = size(h_values)
      series_len = size(simulated_series, 2)
      num_series = size(simulated_series, 1)

      allocate(flag_list(num_series))
      allocate(chart(series_len))

      allocate(ones_list(num_series))
      allocate(zeros_list(num_series))
      ones_list = 1.0d0
      zeros_list = 0.0d0

      do i=1,h_values_len
         h = h_values(i)
         do j=1,num_series
            call control_chart(simulated_series(j, :), expectation, alpha, chart)
            flag_list(j) = flag(chart, h)
         end do
         fp_rate(i) = sum(merge(ones_list, zeros_list, flag_list))/num_series
      end do

    end subroutine false_positive
    
end module cusum
