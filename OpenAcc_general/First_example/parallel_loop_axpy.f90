program mino
  use iso_fortran_env, only: dp=>real64
  implicit none 
  real(dp), allocatable :: a(:), b(:), c(:)  
  integer, parameter :: N = 1000 
  real(dp), parameter   :: alpha = 4.0 * atan(1.0)
  integer :: i 
  allocate(a(N), b(N), C(N)) 
  a = 1.0 
  b = 2.0 
  c = 0.0  
  !$acc parallel loop  copyin(a(:),b(:)) copyin(c(:))  
  do i = 1, N 
     c(i) = c(i) + alpha * (a(i)+b(i)) 
  end do 
  !$acc end parallel loop
  print *, c(1:10) 
end program 
     
