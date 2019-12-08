lazy val core = (project in file("core"))
  .settings(
    organization := "com.github.arzt",
    description := "Scala framework for automatic differentiation and optimization of multivariate functions",
    name := "nabla-core",
    version := "0.0.1-SNAPSHOT",
    scalaVersion := "2.11.12",
    //parallelExecution in Test := false,
    libraryDependencies ++=
      Seq(
        //"com.github" %% "scala-tensor" % "0.0.1-SNAPSHOT",
        "org.nd4j" % "nd4j-native-platform" % "0.9.1",
        "org.nd4j" %% "nd4s" % "0.9.1",
        "org.specs2" %% "specs2-core" % "4.6.0" % "test"
      )
  )
