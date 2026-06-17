# Project-X-Task3<br>
To verify Backpropagation research paper, I considered all 4 digit binary numbers and check whether it is symmetric about center or not.<br>

The src.py file consists of my slution code. At first generate all 16 combinations of 4 digit binary no.s then check if number is same from front and back if yes then stored as 1 else 0.Assign some weights as per the research paper for forward and backward pass. In forward pass to find hidden units values by passing through sigmoid function and finally we calculate Hidden unit's output and then the total system error percentage.<br>

The final ans is nearly correct as both the hidden units in result balance eachother and hence cancel eachother telling us that there is symmetric pattern. Hence proved by  Backpropagation with certain non-zero error.<br>

The results.md contains screenshots of execution.I executed code twice and hence uploaded 2 screenshots because the error percentage changes each time.
