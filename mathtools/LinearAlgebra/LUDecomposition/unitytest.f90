program unitytest
! Program for testing LU and Inverse Process 
! Load LU Decomposition 
use LUDecomposition

  ! Variables
  integer :: n, i

  ! Parameters
  integer, parameter :: maxn = 4
  integer, allocatable  :: indx(:)
  real, allocatable :: matrix(:,:)

  do n = 4 , maxn
    allocate( indx(n), matrix(n, n))
    indx =  (/ (I, I = 1, n) /)
    print *,indx
    call RANDOM_NUMBER(matrix)
    matrix = 100*matrix
    do i = 1, n
      print*,matrix(i,:)
    end do
    call ludcmp_square(matrix, n, indx)
    
    call LUResults(matrix, n)
    
    !do i = 1, n
    !  print*,matrix(i,:)
    !end do 

    call ArrengeMatrix(matrix, indx, n)

    do i = 1, n
      print*,matrix(i,:)
    end do 

    print*, indx

    deallocate( indx, matrix)
  end do


end program unitytest

subroutine LUResults(matrix,n)

  real, intent(inout) :: matrix(n,n)
  real :: L(n,n), U(n,n)
  integer :: i, j, k

  do i = 1, n
    L(i,i) = 1.0
    do j = 1, n
      if (i>j) then
        L(i,j) = matrix(i,j)
        U(i,j) = 0.0
      end if
      if (i<=j) then
        U(i,j) = matrix(i,j)
      end if
      if (i<j) then
        L(i,j) = 0.0
      end if
    end do
  end do

  do i = 1, n
    do j = 1, n
      matrix(i,j) = 0.0 
      do k = 1, n
        matrix(i,j) = matrix(i,j) + L(i,k)*U(k,j)
      end do
    end do
  end do   

end subroutine LUResults

subroutine ArrengeMatrix(matrix, indx, n)

  real, intent(inout) :: matrix(n,n)
  integer, intent(in) :: indx(n)
  real :: temp
  integer :: i, j
  
  do i = n, 1, -1
    if (i /= indx(i)) then
      do j = 1, n 
        temp = matrix(i,j)
        matrix(i,j) = matrix(indx(i),j)
        matrix(indx(i),j) = temp
      end do
    end if
  end do 
end subroutine ArrengeMatrix
