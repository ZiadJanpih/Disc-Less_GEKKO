# Disc-Less_GEKKO
Disc-Less (Discrimination Less) is a Python prototype designed for discrimination-aware data transformation within pre-processing pipelines. This tool focuses on rewriting data transformations to adhere to specific non-discrimination constraints, with a primary emphasis on coverage and fairness. The underlying problem is formulated as a constrained optimization problem of the MINLP (Mixed Integer Non-Linear Programming) type, and a non-linear solver is employed to discover the optimal solution.


# Organization of the repository
* Model.py: This Python file encompasses a simple object-oriented programming (OOP) structure designed for managing the primary objects of the model.
* DiscLess_Gekko: Within this Python file, you'll find all the functions and algorithms crucial to the system.
* Data: This folder houses both the dataset and the input queries file.
* Results: In this folder, you'll find the resultant data.
