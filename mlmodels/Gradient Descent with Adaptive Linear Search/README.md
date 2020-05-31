# Introduction
A simple example of a gradient descent with an adaptive linear search for step size.  The idea is very simple, inorder to accelarate the search for a minimum value the linear search takes in consideration two possible moves:

  * to the left 
  * to the right 
  
of x using the gradient direction. Then, as you can imagine you have g(l) = f(x-lGf), h(d)=f(x+dGf) to decide to which side you move as g and h being linear functions.  
