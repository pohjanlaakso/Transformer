
rm(list=ls())

#install.packages('doParallel')
library(doParallel)
library(foreach)

cores <- detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

# ==============================================================================
# PART 1: THE MATH ENGINE (HELPER FUNCTIONS)
# ==============================================================================

# 1. ReLU Activation (Rectified Linear Unit)
# The "switch" that zeros out negative values
relu <- function(x) {
  x <- as.matrix(x)
  x[x < 0] <- 0
  return(x)
}


# 2. Softmax (Stable Row-wise)
# Converts scores into probabilities. We subtract max(x) for numerical stability.
softmax_matrix <- function(x) {
  # x is a matrix of shape (seq_len, d_model)
  # Apply softmax to each row independently
  y <- t(apply(x, 1, function(row) {
    exps <- exp(row - max(row))
    return(exps / sum(exps))
  }))
  return(y)
}

# 3. Layer Normalization
# Normalizes the vector to have mean 0 and variance 1, then scales/shifts
layer_norm <- function(x, gamma, beta, epsilon = 1e-5) {
  # x: matrix (seq_len, d_model)
  # gamma, beta: vectors (d_model)
  
  mu <- apply(x, 1, mean)      # Mean of each row
  var <- apply(x, 1, var)      # Variance of each row
  
  # Normalize
  x_norm <- (x - mu) / sqrt(var + epsilon)
  
  # Scale and Shift (broadcasting)
  # We transpose to multiply by gamma/beta columns, then transpose back
  out <- t(t(x_norm) * gamma + beta)
  return(out)
}

# 4. Positional Encoding
# Adds information about the order of the GDP quarters
get_positional_encoding <- function(seq_len, d_model) {
  pe <- matrix(0, nrow = seq_len, ncol = d_model)
  position <- 0:(seq_len - 1)
  div_term <- exp(seq(0, d_model - 1, by = 2) * (-log(10000.0) / d_model))
  
  # Apply Sine to even indices, Cosine to odd indices
  pe[, seq(1, d_model, by = 2)] <- sin(outer(position, div_term))
  
  # Handle odd indices (ensure dimensions match if d_model is odd/even)
  if (d_model > 1) {
    pe[, seq(2, d_model, by = 2)] <- cos(outer(position, div_term))
  }
  return(pe)
}

# ==============================================================================
# PART 2: THE TRANSFORMER MODEL ARCHITECTURE
# ==============================================================================

# Initialization: Xavier/Glorot
init_matrix <- function(rows, cols) {
  limit <- sqrt(6 / (rows + cols))
  matrix(runif(rows * cols, -limit, limit), nrow = rows, ncol = cols)
}

# Initialize all learnable weights
init_transformer <- function(d_input = 1, d_model = 16, d_ff = 32, seq_len = 4) {
  weights <- list(
    # 1. Embedding (Project GDP scalar -> d_model vector)
    W_embed = init_matrix(d_input, d_model),
    
    # 2. Attention Mechanism (Single Head for simplicity)
    W_Q = init_matrix(d_model, d_model),
    W_K = init_matrix(d_model, d_model),
    W_V = init_matrix(d_model, d_model),
    W_O = init_matrix(d_model, d_model), # Output projection of attention
    
    # 3. Layer Norm 1
    gamma1 = rep(1, d_model),
    beta1  = rep(0, d_model),
    
    # 4. Feed-Forward Network (FFN)
    W1 = init_matrix(d_model, d_ff),    # Expansion
    b1 = rep(0, d_ff),
    W2 = init_matrix(d_ff, d_model),    # Contraction
    b2 = rep(0, d_model),
    
    # 5. Layer Norm 2
    gamma2 = rep(1, d_model),
    beta2  = rep(0, d_model),
    
    # 6. Final Prediction Head (Project d_model -> GDP scalar)
    W_final = init_matrix(d_model, 1),
    b_final = 0
  )
  return(weights)
}

# The Forward Pass (Inference)
transformer_forward <- function(x, weights, seq_len, return_internals = FALSE) {
  # x: Input vector of length 'seq_len' (e.g., 4 past quarters)
  # Reshape x to matrix (seq_len, 1)
  x_mat <- matrix(x, ncol = 1)
  
  d_model <- ncol(weights$W_embed)
  d_k <- ncol(weights$W_K) # Dimension of Key
  
  # ---------------------------------------------------------
  # STEP 1: EMBEDDING & POSITIONAL ENCODING
  # ---------------------------------------------------------
  # Project inputs to higher dimension
  h <- x_mat %*% weights$W_embed 
  
  # Add Positional Encoding
  pe <- get_positional_encoding(seq_len, d_model)
  h <- h + pe
  
  # ---------------------------------------------------------
  # STEP 2: SELF-ATTENTION
  # ---------------------------------------------------------
  # Create Q, K, V matrices
  Q <- h %*% weights$W_Q
  K <- h %*% weights$W_K
  V <- h %*% weights$W_V
  
  # Calculate Scores (Q * K_transpose / sqrt(d_k))
  scores <- (Q %*% t(K)) / sqrt(d_k)
  
  # Apply Masking (Optional but good for time series): 
  # Ensure we can't see the future. (Lower triangular mask)
  mask <- lower.tri(matrix(1, seq_len, seq_len), diag = TRUE)
  scores[!mask] <- -1e9
  
  # Softmax -> Attention Weights
  attn_weights <- softmax_matrix(scores)
  
  # Weighted Sum of Values
  context <- attn_weights %*% V
  
  # Output Projection
  attn_out <- context %*% weights$W_O
  
  # Add & Norm 1
  h_resid <- h + attn_out
  h_norm1 <- layer_norm(h_resid, weights$gamma1, weights$beta1)
  
  # ---------------------------------------------------------
  # STEP 3: FEED-FORWARD NETWORK (The FFN)
  # ---------------------------------------------------------
  # Linear Expansion
  ffn_hidden <- t(t(h_norm1 %*% weights$W1) + weights$b1)
  
  # ReLU Activation
  ffn_activated <- relu(ffn_hidden)
  
  # Linear Contraction
  ffn_out <- t(t(ffn_activated %*% weights$W2) + weights$b2)
  
  # Add & Norm 2
  h_resid2 <- h_norm1 + ffn_out
  h_norm2 <- layer_norm(h_resid2, weights$gamma2, weights$beta2) 
  
  "In this code the h_norm2 variable is sort of the brain of the model!
  The 8-dimensional thought of the model is later turned into a 2D map."
  
  # ---------------------------------------------------------
  # STEP 4: FINAL PREDICTION
  # ---------------------------------------------------------
  # We take the LAST state (the most recent quarter) for prediction
  last_state <- h_norm2[nrow(h_norm2), , drop = FALSE]
  
  prediction <- last_state %*% weights$W_final + weights$b_final
  
  # test if works! 
  if(return_internals) {
    return(list(prediction = as.numeric(prediction), 
                last_state = as.vector(last_state), 
                attention = attn_weights))
  }
  
  return(as.numeric(prediction))
              
}

# ==============================================================================
# PART 3: TRAINING LOOP (Using Base R 'optim')
# ==============================================================================

# Generate Pseudo GDP Data
set.seed(42)
time_steps <- 100
# Create a trend + seasonal pattern
gdp_data <- 100 + (1:time_steps)*0.5 + sin((1:time_steps)/4)*5 + rnorm(time_steps)
plot.ts(gdp_data)

# Save mean and sd to reverse the scaling later
data_mean <- mean(gdp_data)
data_sd   <- sd(gdp_data)

# Scale data to be roughly between -2 and 2
gdp_data_scaled <- (gdp_data - data_mean) / data_sd

# Prepare Training Data (Sliding Window)
block_size <- 4  # Look back 4 quarters
X_train <- list()
Y_train <- numeric()

for (i in 1:(length(gdp_data_scaled) - block_size)) {
  X_train[[i]] <- gdp_data_scaled[i:(i + block_size - 1)]
  Y_train[i] <- gdp_data_scaled[i + block_size] # Predict the immediate next one
}

# Unflatten weights for 'optim' (optim needs a single vector of parameters)
flatten_weights <- function(weights) {
  unlist(weights)
}

# Reconstruct weights from flat vector
reconstruct_weights <- function(flat_params, original_struct) {
  reconstructed <- list()
  idx <- 1
  
  for (name in names(original_struct)) {
    param <- original_struct[[name]]
    len <- length(param)
    
    if (is.matrix(param)) {
      dims <- dim(param)
      reconstructed[[name]] <- matrix(flat_params[idx:(idx+len-1)], nrow=dims[1])
    } else {
      reconstructed[[name]] <- flat_params[idx:(idx+len-1)]
    }
    idx <- idx + len
  }
  return(reconstructed)
}

# Define Loss Function (Mean Squared Error)
loss_function <- function(flat_params, original_weights, X_list, Y_vec) {
  # Rebuild weight list
  current_weights <- reconstruct_weights(flat_params, original_weights)
  
  total_loss <- 0
  n <- length(Y_vec)
  
  # Loop through data (Production tip: In R, loops are slow, but necessary without matrix libs)
  for (i in 1:n) {
    pred <- transformer_forward(X_list[[i]], current_weights, block_size)
    total_loss <- total_loss + (pred - Y_vec[i])^2
  }
  
  return(total_loss / n)
}

loss_function_parallel <- function(flat_params, original_weights, X_list, Y_vec) {
  current_weights <- reconstruct_weights(flat_params, original_weights)
  n <- length(Y_vec)
  
  # We must export everything needed inside transformer_forward
  sq_errors <- foreach(i = 1:n, 
                       .combine = 'c', 
                       .export = c("transformer_forward", "relu", "softmax_matrix", 
                                   "layer_norm", "get_positional_encoding", 
                                   "block_size", "current_weights")) %dopar% {
                                     
                                     # Each core now knows what 'block_size' and 'current_weights' are
                                     pred <- transformer_forward(X_list[[i]], current_weights, block_size)
                                     (pred - Y_vec[i])^2
                                   }
  
  return(sum(sq_errors) / n)
}

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

# 1. Initialize Model
cat("Initializing Transformer Model...\n")
initial_weights <- init_transformer(d_input=1, d_model=8, d_ff=16, seq_len=block_size)
initial_flat <- flatten_weights(initial_weights)

# 2. Train Model
# We use 'BFGS', a standard numerical optimization algorithm (Quasi-Newton)
# This replaces manual backpropagation.
cat("Training Model (this may take a moment)...\n")
opt_result <- optim(
  par = initial_flat,
  fn = loss_function_parallel,
  original_weights = initial_weights,
  X_list = X_train,
  Y_vec = Y_train,
  method = "BFGS",
  control = list(maxit = 30, trace = 1) # Trace prints progress
)

# Crucial: Stop the cluster when done
stopCluster(cl)

# 3. Get Trained Weights
trained_weights <- reconstruct_weights(opt_result$par, initial_weights)

# 4. Make a Prediction
last_window <- tail(gdp_data_scaled, block_size)
raw_pred <- transformer_forward(last_window, trained_weights, block_size)

next_gdp <- (raw_pred * data_sd) + data_mean

cat("\n-------------------------------------------------\n")
cat("Past 4 Quarters GDP:", round(((last_window * data_sd) + data_mean), 2), "\n")
cat("Predicted Next GDP: ", round(next_gdp, 2), "\n")
cat("Actual Next Value (if known): N/A (Future)\n")
cat("-------------------------------------------------\n")

# ==============================================================================
# PART 4: INTERPRETABILITY & VISUALIZATION
# ==============================================================================

# Extract internal states for all training windows
n_samples <- length(X_train)
hidden_matrix <- matrix(NA, nrow = n_samples, ncol = 8) # ncol = d_model
all_attentions <- list()

for(i in 1:n_samples) {
  internals <- transformer_forward(X_train[[i]], trained_weights, block_size, return_internals = TRUE)
  hidden_matrix[i, ] <- internals$last_state
  all_attentions[[i]] <- internals$attention
}

# 1. PCA: What is the model "thinking"?
pca_res <- prcomp(hidden_matrix, center = TRUE, scale. = TRUE)
var_pc <- round(summary(pca_res)$importance[2, 1:2] * 100, 1)

# Color points by the target GDP value (Red = High, Blue = Low)
color_pal <- colorRampPalette(c("blue", "gray", "red"))(100)
point_cols <- color_pal[as.numeric(cut(Y_train, breaks = 100))]

par(mfrow=c(1,2)) # Side-by-side plots

# Plot A: PCA Manifold
plot(pca_res$x[,1], pca_res$x[,2], col="black", bg=point_cols, pch=21, cex=1.5,
     main="PCA: Hidden State Space",
     xlab=paste0("PC1 (", var_pc[1], "%)"), ylab=paste0("PC2 (", var_pc[2], "%)"))
grid()
legend("topright", legend=c("High GDP", "Low GDP"), pt.bg=c("red", "blue"), pch=21, cex=0.8)

# 2. Attention: What is the model "looking at"?
# We take the average attention across the entire dataset
avg_attn <- Reduce("+", all_attentions) / length(all_attentions)
# We care about the last row: How the "current" quarter looks at the past
final_attn_row <- avg_attn[block_size, ]

# Plot B: Attention Weights
barplot(final_attn_row, names.arg = paste0("T-", (block_size-1):0),
        col = "steelblue", main = "Average Attention Focus",
        xlab = "Past Quarters", ylab = "Weight/Importance")

cat("\nAnalysis Complete.\n")
cat("1. Check the PCA plot: If points form a smooth line/curve sorted by color,\n")
cat("   the model has linearized the economic trend internally.\n")
cat("2. Check the Barplot: This shows which of the 4 quarters the model\n")
cat("   actually relies on to make its prediction.\n")

"If attention weights are 0.25, 0.25, 0.25, 0.25 the model is acting like a simple
moving average. If  the weight for the most recent quarter is high, the model is
momentum. "