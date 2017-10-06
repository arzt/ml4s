package ml4s

package fun

object Softmax {

  def apply(in: Int): Differentiable = DivFun(Exp(in), Sum(Exp(in))) ∘ Repeat(in, 2)

  def apply(fun: Differentiable): Differentiable = Softmax(fun.out) ∘ fun

}
