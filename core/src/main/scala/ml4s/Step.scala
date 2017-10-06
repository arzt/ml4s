package ml4s

case class Step(x: ND, grad: ND, δgrad: Double) {
  override def toString: String = f"  δ∇=$δgrad%2.12f"
}