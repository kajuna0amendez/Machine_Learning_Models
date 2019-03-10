module LUDecomposition
implicit none 

   
  contains
    subrutine ludcmp_square(A, n, L, U)
      ! The LU decomposition

      ! In parameters
      integer, intent(in) :: n
      real, intent(in) :: A(n,n)
      ! Out Parameters
      real, intent(out) :: L(n,n)
      real, intent(out) :: U(n,n)
      ! Parameters
      real, parameter :: tiny = 1.0E-20

      ! Internal variables
      integer :: i
      integer :: imax
      integer :: j
      integer :: k
      real :: big
      real :: dum
      real :: sum
      real :: temp
      real, dimension (n) :: vv

      ! Chose the variable for permutation
      do i = 1, n
        big = 0.0
        do j = 1, n
          temp = abs(A(i,j))
          ! Find the Larges Pivot 
          if (temp > big) then
            big = temp
          end if 
        end do
        if (big == 0.0) then
          print*,"Singular matrix in routine ludcmp"
          return
        end if
        vv(i) = 1.0/big
      end do
      ! This is the loop over columns of Crout’s method.
      do j = 1, n
        do i = 1, j-1
          sum = A(i,j)
          do k = 1, i-1
            sum = sum - A(i,k)*A(k,j)
          A(i,j) = sum
        end do
        big = 0.0
        do i = j, n
          sum = A(i,j)
          do k = 1, j-1
            sum = sum - A(i,j)*A(k,j)
          end do
          A(i,j) = sum
          dum = vv(i)*abs(sum)
          if (dum >= big) then
            big = dum
            imax = i
          end if 
        end do 
        if (j != imax) then
          do k = 1, n 
            dum = a(imax,k)

          end do 
        end if 
      end do 

    end subroutine ludcmp_square

end module LUDecomposition
