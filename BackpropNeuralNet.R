## Neural Network Project using Backpropagation.

# Top level function

NeuralNet <- function(training, testing, validation, learning.rate = 0.1, momentum.coef = 0.5, neuron.count.per.layer = c(2,3,1), epochs = 10000, ShowPlot = TRUE){
  
  ## Read in Data files
    # Assuming the first three arguments are file paths as character stings(e.g. "testing.txt"), import the file as a data.frame and convert to a matrix.
      training <- data.matrix(read.table(training))
      testing <- data.matrix(read.table(testing))
      validation <- data.matrix(read.table(validation))
  
  
  ## Initial Assignments
    # Weights and bias (threshold) values
      layer.weights <- list()    # Initialize the weights object as a blank list
      bias.weights <- list()     # Initialize the biases object as a blank list
  
  ## Varying layer counts, usually used for indexing purposes in loops.
   # Determines the total number of layers by counting the number of elements in the neuron.count.per.layer argument (6th argument)
    n.layers <- length(neuron.count.per.layer)
   # Creates a two-element vector, containing the number of neurons at the input layer, [1], and the output layer [(total number of layers)]
    in.out.n.sum <- sum(c(neuron.count.per.layer[1], neuron.count.per.layer[n.layers]))
   # A simple calculation to determine the number of hidden layers, all layers minus the number of input-output layers, which is always 2.
    n.hidden.layers <- n.layers - 2
  
  # Errors Objects
  # Initializing an "empty" list with NULL elements under each pre-assigned list index.
    Errors <- list()
    Errors$Training <- NULL
    Errors$Testing <- NULL
    Errors$Validation <- NULL
  
  ## User-error Checking and Correction
    # If you only entered one layer, adds a second layer with one output neuron. May or may not be useful.
      if (n.layers == 1){
        neuron.count.per.layer <- c(neuron.count.per.layer,1)
      }
    # If you designated a number of neurons on the input-output layers that is inconsistent with the training data, this assumes you have one output, 
    #  and assigns the rest to input based on the training set dimensions.
      if ((in.out.n.sum  > dim(training)[2]) || (in.out.n.sum  < dim(training)[2])){
        neuron.count.per.layer <- c((dim(training)[2]-1), 1)
      }
    # If you assigned an appropriate number of neurons, but only to one layer, one neuron is moved to the output layer.
     if (neuron.count.per.layer[1] ==  dim(training)[2]){
        neuron.count.per.layer <- c((neuron.count.per.layer-1), 1)
      }
  
  ## Initiation of random neuron weights
    # Each list item layer.weights[[i]] contains a weight matrix that transforms outputs from the (i-1)th layer into input sums (h, the local field) for the i-th layer.
    # Also creates bias weights in separate list, which are threshold values for each neuron treated as an extra weight.
  
    # Weight matrices signify connection layers rather than neuron layers, used for transformations. Thus, there will always be one less number of weight matrices than neuron layers.
      # It is assumed that no transformations take place for activation of the input layer. Input values equal the local field values, h, for each input neuron.
      # A similar assumption is made at the output layer. There is no weight associated with an output neuron upon evaluation, but rather gives value between 0 and 1 defined by the local field applied to the sigmoid function.
    
  
    ## First, if there are hidden layers in the system, all weights will be initialized in sequence, layer by layer.
  
      if (n.hidden.layers > 0){
    
        for (i in 1:(n.hidden.layers+1)){
          
          layer.weights[[i]] <- matrix(runif((neuron.count.per.layer[i]*neuron.count.per.layer[i+1]), min= -0.1, max = 0.1), nrow = neuron.count.per.layer[i], ncol = neuron.count.per.layer[i+1])
          bias.weights[[i]] <- runif(neuron.count.per.layer[i+1], min = -0.1, max = 0.1)
        }
    ## If there are no hidden layers, create only 1 weight matrix directly between the input and output neurons.
        
      }else{
        layer.weights[[1]] <- matrix(runif((neuron.count.per.layer[1]*neuron.count.per.layer[2]), min= -0.1, max = 0.1), nrow = neuron.count.per.layer[1], ncol = neuron.count.per.layer[2])
        bias.weights[[1]] <- runif(neuron.count.per.layer[i+1], min = -0.1, max = 0.1)
      }
  
  ## Sort Validation set.
    # This operation assumes that input and output values are all in sequence (input 1 , input 2, ..., input n, output 1, output 2, ..., output n), one row per set.
    # Assuming the number of inputs equals the number of input neurons, the first (1 to 'n-inputs') elements per row are separated as an input matrix.
    # Likewise for outputs, the rest of each row is separated as an output matrix.
    # Unlike the Training and Testing sets, the Validation set is not randomized, because it is only evaluated once, without any change to the weights.
      validation.inputs <- validation[,(1:neuron.count.per.layer[1])]
      validation.outputs <- validation[,-(1:neuron.count.per.layer[1])]
  
  ## This section trains the neural net and tests it with the training and testing sets repsectively.
    # Each epoch runs through the entirety of the training set and the testing set once each. Each epoch is another iteration through the entire sets.
  
    # i3 is initialized to 1, and it will run as an epoch counter in the while-loop. 
    # The loop will terminate after a set number of epochs, supplied as an agrument to the parent function.
      i3 <- 1 
  
  while (i3 != epochs+1){  # Given the placement of the count update (at the end), the termination value is set to 1 greater than the maximum number of epochs.
    
    # Randomizes order of sets for each epoch.
    # This ensures that the Neural Net doesn't learn to approximate the parent function based on the order of the sets.
      training <- training[sample(nrow(training)),]
      testing <- testing[sample(nrow(testing)),]
    
    ## Sorting Input columns from output columns.
      # This operates with the same sorting scheme used for the Validation set, see above (74-80).
        training.inputs <- training[,(1:neuron.count.per.layer[1])]
        training.outputs <- training[,-(1:neuron.count.per.layer[1])]
        testing.inputs <- testing[,(1:neuron.count.per.layer[1])]
        testing.outputs <- testing[,-(1:neuron.count.per.layer[1])]
    
    # All Error values are initially set to 0, but they will be changed once the net has been evaluated at least once.
    # However, the Validation error will only be updated one time, at the very end of training.
      training.error <- 0
      testing.error <- 0
      validation.error <- 0
    
    # These two values are used for the implementation of the momentum coefficient, which incorporates the weight change of the previous epoch into the current one.
    # Likewise to the initiation of the errors, these will change once the Neural Net has been evaluated at least once. 
    # These variables correspond to lists that occur later in the script, weightsprev and biasprev, which hold the previous changes of the entire Net, rather than just one connection layer.
      weight.change <- 0
      bias.change <- 0
    
    ## This function runs one pattern (one row) through the entire Neural Net.
      # The argument 'p' is a vector of input values given to the input layer.
      # 'i2' is used as a layer counter, signifying the number of connection layers, hence (n.layers - 1)
      # Note that the sigmoid list is initialized with the input pattern. This allows the input values to be passed to the first hidden (or output) layer only according to the weights that connect them.
      # 'h' signifies the local field of the next neuronal layer, calculated by matrix multiplication, transforming the inputs from the previous layer into sums according to connection weights from the previous layer.
      # In the next appearance of 'sigmoid', all local field values are now translated into sigmoid values using the pre-determined sigmoid function
        # These values will be used as inputs for the next iteration in the loop.
      # 'pattern' is used as separate variable for the last sigmoid vector obtained (the output sigmoids). 
        # At the end of the loop, this vector is compared to the desired values given as 'training.outputs', 'testing.outputs', or 'validation.outputs'.
  
        RunNet <- function(p){     
          i2 <- 0
          sigmoid <<- list(p)
          pattern <- p
          while (i2 != (n.layers-1)){
            i2 <- i2+1
            h <- (pattern %*% layer.weights[[i2]])+bias.weights[[i2]]
            sigmoid[[i2+1]] <<- as.numeric(1/(1+exp(-h)))
            pattern <- t(matrix(sigmoid[[i2+1]]))
          }
          return(pattern)
        }
  
  ### Backpropagation 
    # As mentioned previously, weightsprev and biasprev are lists that hold the entirety of weight changes used in the previous epoch.
    # Each will have the same exact structure as layer.weights and bias.weights.
      weightsprev <- list()
      biasprev <- list()
    
      for (u in (1:(length(neuron.count.per.layer)))){
        weightsprev[[u]] <- 0
        biasprev[[u]] <- 0
      }
    ## From here, there is a nested loop, where first the Neural Net is run with each training pattern, and then the weights are updated after each pattern.
      # First, a 'delta' list is initialized, which will hold the delta values associated with each layer for calculations of weight-updates (I will refer to them as 'influence').
      # Then, each training set is introduced, run through the Neural Net via 'RunNet()', and the last (output) values are stored in 'output', which is what 'RunNet()' eventually returns.
        for(w in (1:length(training.outputs))){
          delta <- list()
          output <- RunNet(training.inputs[w,])
          
      # Here, the weights are updated using derived equations that calculate the influence of each hidden neuron and output neuron on the outputs obtained.
      # The equations associated with this section are quite complicated. See the document provided for a clearer presentation of the calculations occuring here.
          
          for(i in (n.hidden.layers+1):1){
            if (i == (n.hidden.layers+1)){
              # If we are at the output layer, compare the outputs obtained to the desired outputs and calculate the influence of the output neurons.
              delta[[i]] <- 2*output*(1-output)*(training.outputs[w] - output)
            }else{
              # If we are at a hidden layer, calculate the influence of the hidden neurons based on the influence from the previous layer.
              delta[[i]] <- sigmoid[[i+1]]*(1-sigmoid[[i+1]])*(t(matrix(delta[[i+1]])) %*% t(layer.weights[[i+1]]))
            }
          # Once all influence values have been obtained for the current layer, calculate the weight changes for the current layer.
          # Note the 'weightsprev' and 'biasprev' variables show up here, modified by a coefficient (supplied as an agrument, 'momentum.coef'). These signify the "momentum" from the previous epoch.
            weight.change <- t((learning.rate * (matrix(delta[[i]]) %*% t(matrix(sigmoid[[i]]))))) +  momentum.coef*weightsprev[[i]]
            bias.change <-  (learning.rate * delta[[i]]) +  momentum.coef*biasprev[[i]]
          
          # Once the weight changes have been supplied for this connection layer, store them for use in the next epoch as momentum.
            weightsprev[[i]] <- weight.change
            biasprev[[i]] <- bias.change
          
          # Finally update the weights and the bias values for the current layer.
            layer.weights[[i]] <- layer.weights[[i]] + weight.change
            bias.weights[[i]] <- bias.weights[[i]] + bias.change
          }
          # Calcuate the error by the sum of square differences over all output neurons and add this to the value obtained in the previous epoch.
            training.error <- training.error + sum((output - training.outputs[w])^2) 
        }
      
      # After both loops above have completed, having run through 1 epoch (the entire training set), take the mean of the sum-of-squares error over the number of output neurons and the number of training sets.
      # This value is used to ultimately judge the performance of the Neural Net.
        training.error <- (training.error/neuron.count.per.layer[length(neuron.count.per.layer)])/length(training.outputs)
    
    ## Apply Testing patterns to net.
      # This is a simple loop that runs each testing pattern through the net, and accumulates the sum-of-squares error.
        for(z in 1:(length(testing.outputs))){
          testing.output <- RunNet(testing.inputs[z,])
          testing.error <- testing.error + (testing.output - testing.outputs[z])^2 
        }
      # Same as above for training error, the mean sum-of-squares error over output neurons and testing sets.
        testing.error <- (testing.error/neuron.count.per.layer[length(neuron.count.per.layer)])/length(testing.outputs)
    
    ## Record and print Errors
     # For each epoch, prints the epoch number, the current training error and the current testing error as mean sum-of-squares error values.
      cat("\n epoch: ", i3, "   Training Error:", training.error, "   Testing Error:", testing.error)
      
     # The error at each epoch is compiled into one long vector, which will be used later for plotting the errors across epochs
      Errors$Training <- c(Errors$Training,training.error)
      Errors$Testing <- c(Errors$Testing,testing.error)
      
  # Finally, update the epoch number, and do it all over again!  
    i3 <- i3 + 1
  }
  
  ## Apply Validation patterns to net.
   # This is applied in the same exact way as the Testing set, but only one time, after training has completed.
      for(z in 1:(length(validation.outputs))){
        validation.output <- RunNet(validation.inputs[z,])
        validation.error <- validation.error + (validation.output - validation.outputs[z])^2
      }
      Errors$Validation <- (validation.error/neuron.count.per.layer[length(neuron.count.per.layer)])/length(validation.outputs)
  # Removes extraneous variables in the global environment.
    rm("sigmoid", pos = .GlobalEnv)
    rm("delta")
  
  # A variable that contains the final weight matrices used, the final bias weights used, the testing error minimum, and the Validation error at the end.
    results <- list("Final Weights" = layer.weights, "Bias Weights" = bias.weights, "Testing Minimum" = min(Errors$Testing), "Validation Error" = Errors$Validation)
  
  # Do you want to see a plot of the errors across all epochs? Well, if you supplied "TRUE" to 'ShowPlot' in the parent function, a plot will appear!
    if (ShowPlot == TRUE){
      plot(Errors$Training, type = "l", xlab = "Epoch", ylab = "Mean Squares Error", main = "Testing Error & Training Error v. Epoch")
      lines(Errors$Testing, col = 2)
      legend("topright", legend = c("Training Error", "Testing Error"), lty = c(1,1), lwd = c(2.5,2.5), col = c(1,2), merge = TRUE, inset = 0.05)
    }
  cat("\n \n")
  return(results)
}