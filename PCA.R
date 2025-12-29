
source('simple_transformer.R')

# ==============================================================================
# PART 4: INTERPRETABILITY (OPENING THE BLACK BOX)
# ==============================================================================

# ---------------------------------------------------------
# A. Feature Extraction
# ---------------------------------------------------------
# We need a modified forward pass that returns the "Hidden State" 
# (the vector just before the final prediction) instead of the prediction itself.
get_hidden_state <- function(x, weights, seq_len) {
  # ... (Repeat the forward pass steps exactly as before) ...
  # For brevity, we assume the same logic as 'transformer_forward' 
  # but we stop before 'W_final'.
  
  x_mat <- matrix(x, ncol = 1)
  d_model <- ncol(weights$W_embed)
  d_k <- ncol(weights$W_K) 
  
  # 1. Embed + Pos
  h <- x_mat %*% weights$W_embed 
  pe <- get_positional_encoding(seq_len, d_model)
  h <- h + pe
  
  # 2. Attention
  Q <- h %*% weights$W_Q
  K <- h %*% weights$W_K
  V <- h %*% weights$W_V
  scores <- (Q %*% t(K)) / sqrt(d_k)
  mask <- lower.tri(matrix(1, seq_len, seq_len), diag = TRUE)
  scores[!mask] <- -1e9
  attn_weights <- softmax_matrix(scores) # <--- Valuable for Method B
  context <- attn_weights %*% V
  attn_out <- context %*% weights$W_O
  h_resid <- h + attn_out
  h_norm1 <- layer_norm(h_resid, weights$gamma1, weights$beta1)
  
  # 3. FFN
  ffn_hidden <- t(t(h_norm1 %*% weights$W1) + weights$b1)
  ffn_activated <- relu(ffn_hidden)
  ffn_out <- t(t(ffn_activated %*% weights$W2) + weights$b2)
  h_resid2 <- h_norm1 + ffn_out
  h_norm2 <- layer_norm(h_resid2, weights$gamma2, weights$beta2)
  
  # Extract the state of the LAST time step (the one used for prediction)
  last_state <- h_norm2[nrow(h_norm2), ]
  
  # Return both the state and the attention weights
  return(list(state = last_state, attention = attn_weights))
}

# Collect hidden states for all training examples
n_samples <- length(X_train)
hidden_matrix <- matrix(NA, nrow = n_samples, ncol = ncol(trained_weights$W_embed))
# We will also store the actual GDP value to color the plot
actual_values <- Y_train 

cat("Extracting internal representations...\n")
for(i in 1:n_samples) {
  out <- get_hidden_state(X_train[[i]], trained_weights, block_size)
  hidden_matrix[i, ] <- out$state
}

# ---------------------------------------------------------
# B. PCA Visualization
# ---------------------------------------------------------
# Perform PCA on the hidden states
pca_result <- prcomp(hidden_matrix, center = TRUE, scale. = TRUE)

# Calculate variance explained (to see how much info is in 2D)
var_explained <- round(summary(pca_result)$importance[2, 1:2] * 100, 1)

# Plot
# We color the points by the ACTUAL GDP value of that period.
# If the model works, similar colors should group together.
palette_func <- colorRampPalette(c("blue", "white", "red"))
colors <- palette_func(100)[as.numeric(cut(actual_values, breaks = 100))]

plot(pca_result$x[, 1], pca_result$x[, 2], 
     bg = colors, pch = 21, cex = 1.5,
     main = "PCA of Transformer Internal States",
     xlab = paste0("PC1 (", var_explained[1], "%)"),
     ylab = paste0("PC2 (", var_explained[2], "%)"))
grid()
legend("topright", legend=c("Low GDP", "High GDP"), pt.bg=c("blue", "red"), pch=21)

cat("\nInterpretation:\n")
cat("Look at the PCA plot. If the dots transition smoothly from Blue (Low GDP) \n")
cat("to Red (High GDP), the model has successfully mapped the magnitude of GDP \n")
cat("into its internal vector space.\n")

# ---------------------------------------------------------
# C. Attention Analysis (The "Why")
# ---------------------------------------------------------
# PCA shows "State", Attention shows "Focus".
# Let's look at the attention weights for the very last prediction.
last_run <- get_hidden_state(last_window, trained_weights, block_size)
last_attn <- last_run$attention

# The last row represents the attention of the 'Current Quarter' looking back.
current_quarter_focus <- last_attn[block_size, ]

cat("\nAttention Weights (Last Prediction):\n")
names(current_quarter_focus) <- paste0("Q-", (block_size-1):0)
print(round(current_quarter_focus, 3))

cat("\nInterpretation:\n")
cat("If 'Q-0' (most recent) is highest, the model relies on immediate momentum.\n")
cat("If 'Q-3' is high, the model might be looking at year-over-year seasonality.\n")