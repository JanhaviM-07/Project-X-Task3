Learning representations by back-propagating errors is research paper  by David E. Rumelhart, Geoffrey E. Hintont & Ronald J. Williams.

<b>Central Claim</b><br>
The Central claim is that  a generalized rule i.e by backpropagation we can successfully optimize multi-layer neural networks containing internal "hidden" units. Prior, simpler methods like the perceptron convergence procedure were fundamentally limited because they could only learn linear decision boundaries and could not determine representations for hidden layers where the target states are unobserved.This was a big deal because Backprop unlocked non-linear learning. 
In perceptrons, there are 'feature analysers' between the input and output that are not true hidden units because their input connections are fixed by hand, so their states are completely determined by the input vector: they do not learn representations.

<b>Core Architecture & Algorithm</b><br>
The simplest form of the learning procedure is for layered networks which have a layer of input units at the bottom any number of intermediate layers and a layer of output units at the top.A unit has a real-valued output, y, which is a non-linear function of its total input:<br>
<br>
<img width="100" height="50" alt="Screenshot 2026-06-15 234025" src="https://github.com/user-attachments/assets/62322919-665a-4a0f-94c1-036bc929120e" />

i.e.  yj=f(xj)

where xj is the total weighted input sum: <br>
<img width="140" height="50" alt="Screenshot 2026-06-15 234006" src="https://github.com/user-attachments/assets/92d53d82-b3d7-44c7-95cd-4c9019cc8e82" />

<br>
Output Layer: Sigmoid-activated units outputting continuous values between 0 and 1.
<br><br><b>The Algorithm to Implement:</b><br>
<b>Forward Pass:</b> Propagate inputs through successive layers to compute the actual state ‘y’ of every unit.To evaluate the total error using the sum-of-squared errors metric over all training cases’c’ and output units ‘j’:
<br><img width="150" height="50" alt="Screenshot 2026-06-15 234040" src="https://github.com/user-attachments/assets/bbfa13a9-3cda-4f04-9958-ac014aeb6a0f" />
<br>
<b>Backward Pass:</b>Compute partial derivatives of total input recursively from the output layer back to the input layer using the chain rule:
<br>For an output unit:<br>
<img width="150" height="40" alt="Screenshot 2026-06-16 000139" src="https://github.com/user-attachments/assets/1a1d5d93-d407-42b9-85d9-317c8dca36da" /> ;
 <img width="180" height="40" alt="Screenshot 2026-06-16 000149" src="https://github.com/user-attachments/assets/2b8ab419-9813-41c9-b1ab-01c80dd30297" />
<br><img width="190" height="40" alt="Screenshot 2026-06-16 000203" src="https://github.com/user-attachments/assets/6036825a-dae3-4a71-ae7d-71caa4c64ade" />;
<img width="200" height="40" alt="Screenshot 2026-06-16 000211" src="https://github.com/user-attachments/assets/07c1a9e8-27c8-4b56-8d5b-dbc770340082" />
<br>
For a hidden unit: <br>
<img width="190" height="40" alt="Screenshot 2026-06-16 000203" src="https://github.com/user-attachments/assets/6036825a-dae3-4a71-ae7d-71caa4c64ade" />;
<img width="170" height="50" alt="Screenshot 2026-06-16 001838" src="https://github.com/user-attachments/assets/878ca446-954e-486f-a72c-f2bbf3b88f66" />

<br>
<b>Weight Adjustments:</b> Apply gradient descent with a momentum parameter to update connections after every epoch or pattern step:
<br><img width="240" height="40" alt="Screenshot 2026-06-16 001309" src="https://github.com/user-attachments/assets/28ecd350-f762-457a-9470-a2b21aba402b" />
<br>


<b>3. Datasets, Evaluation Metrics</b><br>

<b>The Dataset:</b> A vector of binary values i.e if  6 inputs, then 2^6 = 64 total possible binary combinations. To confirm that the implementation matches the paper's benchmarks, I have chosen the Symmetry Detection Problem i.e.   to check if input number is perfectly symmetric around its about middle or not by backpropagation. 
