# Programmatic edit test via fs.write
module TestModule

export add_numbers, multiply_numbers, square, cube, demo

add_numbers(x, y) = x + y
multiply_numbers(x, y) = x * y
square(x) = x^2
cube(x) = x^3

# New demo block inserted programmatically via fs.write
function demo()
    println("Running TestModule.demo()")
    println("add_numbers(1,2)=", add_numbers(1,2))
    println("multiply_numbers(2,4)=", multiply_numbers(2,4))
    println("square(5)=", square(5))
    println("cube(2)=", cube(2))
end

println("TestModule updated again via fs.write")
println("add_numbers(2,3)=", add_numbers(2,3))
println("multiply_numbers(2,3)=", multiply_numbers(2,3))
println("square(4)=", square(4))
println("cube(3)=", cube(3))

end

