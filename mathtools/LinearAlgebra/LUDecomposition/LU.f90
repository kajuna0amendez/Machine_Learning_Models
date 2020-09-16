module LUDecomposition
implicit none 
   
  contains
    subroutine ludcmp_square(A, n, indx)

      ! The LU decomposition
      ! Return the LU decomposition in A after all alpah_ii = 1
      ! Thus Rule 
      ! beta_ij  = a_ij with i<j
      ! alpha_ij = a_ij with i>=j
      !
      ! Steps of the Algorithm
      ! 1.- Find the largest absolute element in the row to be used as  
      ! 2.- if big == 0.0 you have a singular matrix return with error
      ! 3.- if not singularity save the scalling vv(i) = 1.0/big
      ! 4.- Loop over the betas 
      ! 5.- Loop over the betas and look if you have a new max pivote
      ! 6.- Move the max row pivote upwards
      ! 7.- Use TINY number if division by zero arises 
      ! 8.- Divide by pivote element


      ! In parameters
      integer, intent(in) :: n
      real, intent(inout) :: A(n,n)
      ! Out Parameters
      integer, intent(out) :: indx(n)
      ! Parameters
      real, parameter :: TINY = 1.0E-30

      ! Internal variables
      ! Indexing
      integer :: i
      integer :: j
      integer :: k
      ! max index pivot
      integer :: imax
      ! Internal variables for the Crout's method
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
          if (temp > big) then
            big = temp
          end if 
        end do
        if (big == 0.0) then
          print*,"Singular matrix in routine ludcmp"
          return
        end if
        ! Save the Scaling
        vv(i) = 1.0/big
      end do
      ! This is the loop over columns of Crout’s method.
      ! For the upper triangular part of A(i,j)=beta_ij with i<=j
      do j = 1, n
        do i = 1, j-1
          sum = A(i,j)
          do k = 1, i-1
            sum = sum - A(i,k)*A(k,j)
          end do 
          A(i,j) = sum
        end do
        ! Start the search for the largest pivot element
        big = 0.0
        imax = j
        do i = j, n
          sum = A(i,j)
          do k = 1, j-1
            sum = sum - A(i,k)*A(k,j)
          end do
          A(i,j) = sum
          dum = vv(i)*abs(sum)
          if (dum >= big) then
            big = dum
            imax = i
          end if 
        end do 
        ! Interchage Rows a classic to mantain stability getting 
        ! The Largest absolute value in the choosen pivot
        if (j /= imax) then
          do k = 1, n 
            dum = A(imax,k)
            A(imax,k) = A(j,k)
            A(j,k) = dum
          end do
          ! Interchange the scale factor
          vv(imax) = vv(j)
        end if
        indx(j) = imax
        ! Solving some issues with zero pivots
        !
        if (A(j,j) == 0) then
          A(j,j) = TINY
        end if
        ! Divide by the pivot element.
        if (j /= n) then
          dum = 1.0/A(j,j)
          do i = j+1, n
            A(i,j) = A(i,j)*dum ! Division 
          end do
        end if
      end do 

    end subroutine ludcmp_square

end module LUDecomposition
