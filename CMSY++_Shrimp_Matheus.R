##############################################################################################
# CMSY++ implementation Froese et al. (2021)
# Shrimp Brazilian Fisheries 
# Inclusion of life history parameters to estimate informative rmax (Cortes, 2016)
# Inclusion of zBRT (Zhou et al. 2017)  and CMSY Neural network to estimate depletion B/k
# Sensitivity analysis to evaluate priors distributions´
# Combining final results of reconstructed time series (Silva and Freire)
# Created and Modified by Matheus L. (Silvamls) 11/2025
##############################################################################################

#---------------------------------------  Stock units ------------------------------------------------------
#•	Seabob shrimp in the North (Seabob_N), Northeast (Seabob_NE), Southeast (Seabob_SE), South (Seabob_S) regions.
#•	Brown shrimp in the North (Brown_N) and Northeast (Brown_NE).
#•	Pink shrimp in the Southeast (Pink_SE) and South (Pink_S).
#•	White shrimp in the North (White_N), Northeast (White_NE), Southeast (White_SE), and South (White_S). 
# **Maranhão state as North region**
#-----------------------------------------------------------------------------------------------------------

#@@packages..
# Run if you don't already have devtools installed
#install.packages("devtools")
# Run once devtools is successfully installed
#devtools::install_github("cfree14/datalimited2",force = TRUE) #to run zBRT Zhou method 

library(devtools)
library(datalimited2)
library(mgcv)
library(dplyr)
library(tidyr)
library(plyr)
library(tibble)
library(keras)
library(furrr)
library(future)
library(purrr)
library(readr)
library(ggplot2)
library(R2jags)  # Interface with JAGS (download also: https://sourceforge.net/projects/mcmc-jags/)
library(coda)
library(gplots)
library(mvtnorm)
#library(snpar)
library(neuralnet)
library(conicfit)
library(geobr)
library(sf)
library(rnaturalearth)
library(caret)
library(foreach)
library(doParallel)
library(rlang)   # Helpers (e.g., %||%)
library(stringr)
library(patchwork)
library(scales)
library(glue)

#defining workspace......
rm(list = ls())
graphics.off()
dir="C:/Matheus/Universidade/Doutorado/Avaliação Shrimp_CMSY" 
setwd(dir)
#--------------------------------------------------------------------------------------------------------------
#reading data...
ct<-read.csv("shrimp_catch.csv", dec = ".",sep = ",")#Reconstructed catches (Freire et al.,.. Silva et al.,..)

lf=read.csv("LifeHistoryParameters.csv", sep=",", dec=".", #Life history table
                 fileEncoding = "Latin1", #keep accentuation
                 check.names = F,  #don't check variable names
                 header = TRUE, 
                 stringsAsFactors = F) #keep strings as strings

#Load the trained nets and constants produced by CMSY++
#    (ffnn.bin must be in your working directory )
load("ffnn.bin")   # loads nn.endbio and normalization bounds into the workspace

#-------------------------------------------------------------------------------------------------------------



#-----------------------------
#creating a map of stocks...
#-----------------------------
# Latin America Map
latin_america <- ne_countries(scale = "medium", continent = "South America", returnclass = "sf")

# Brazilian states shapefile
br_states <- read_state(year = 2020)

#take only coastal states
coastal_states <- br_states %>%
  mutate(region = case_when(
    abbrev_state %in% c("AC", "AM", "RO", "RR", "TO", "GO", "MT", "DF", "MS", "MG") ~ "Other",
    abbrev_state %in% c("AP", "PA", "MA") ~ "North",
    abbrev_state %in% c("CE", "RN", "PB", "PE", "AL", "SE", "BA", "PI") ~ "Northeast",
    abbrev_state %in% c("ES", "RJ", "SP") ~ "Southeast",
    abbrev_state %in% c("PR", "SC", "RS") ~ "South",
    TRUE ~ "Other"
  ))

# Plot map
p1<- ggplot() +
  #Latin America
  geom_sf(data = latin_america, fill = "grey90", color = "white") +
  #rest of states
  geom_sf(data = filter(coastal_states, region == "Other"), fill = "grey80", color = "white") +
  #Coastal states 
  geom_sf(data = filter(coastal_states, region != "Other"), aes(fill = region), color = "white") +
  #Letters for each region
  #geom_text(data = region_labels, aes(x = x, y = y, label = letter),
  #          color = "black", size = 4, fontface = "bold") +  
  geom_hline(yintercept = 0 , linetype=2,color = "grey30")+
  scale_fill_manual(values = c("North" = "#66c2a5",
                               "Northeast" = "#fc8d62",
                               "Southeast" = "#8da0cb",
                               "South" = "#e78ac3")) +
  labs(fill = "Stocks", x="Longitude",y="Latitude") +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank(),
        plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"),
        axis.text = element_text(size = 12),
        axis.ticks = element_line(color = "grey80"),
        axis.title = element_text(size = 12),
        panel.grid = element_line(color = "grey90")) +
  #Limits
  coord_sf(
    xlim = c(-80, -37),  # longitudes –80°  20° 
    ylim = c(-50, 10)    # latitudes  –60°  0° 
  ) +
  #Text
  annotate("text", x = -50, y = -12, label = "Brazil", color = "grey30", size = 5)+
  annotate("text", x = -42, y = 0.7, label = "Equator Line", color = "grey20", size = 3.5)

p1

ggplot2::ggsave("Stocks_map.png", plot = p1, device = "png", units = "cm", width = 30, height = 18, dpi = 300,bg = "transparent")
#-----------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------
#Catch projections (2016-2025)- Alternative scenarios 
#Artificial Neural Networks - Feed Forward networks
#different testing sets
#Different units, activation functions, optimizer, layers..
#---------------------------------------------------------

# Criar coluna 'stock'
ct$stock <- paste(ct$category, ct$region, ct$source, sep = "_")

set.seed(NULL)

#Hyperparameters grid
unit_layer <- c(16,32,64)
act_fun <- c("relu","tanh")
drop_rate <- c(0.2,0.3)
opt_fun <- c("adam","rmsprop")
layers <- c(2,3,4)

hyper_grid <- expand.grid(unit_layer = unit_layer,
                          act_fun = act_fun,
                          drop_rate = drop_rate,
                          opt_fun = opt_fun,
                          layers = layers,
                          stringsAsFactors = FALSE)

#Random sampling combinations
n_trials <- 20
n_trials <- min(n_trials, nrow(hyper_grid))
trials <- hyper_grid[sample(nrow(hyper_grid), n_trials), ]

# Rolling-origin params
n_folds <- 3
test_window <- 3

# Parallel processing parameters
n_cores <- parallel::detectCores() - 1
cl <- makeCluster(n_cores); registerDoParallel(cl)

# Hyperparameter evaluation function
# This function evaluates a set of hyperparameters (hp_row) using time-series
# cross-validation, computing Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
eval_hp <- function(hp_row, x_all, y_all, n_folds = 3, test_window = 3, val_frac = 0.15) {
  
  # Initialize vectors to store metrics for each fold
  maes <- c(); rmses <- c()
  n <- length(x_all)
  
  # Loop through each fold of the time-series cross-validation
  for (i in seq_len(n_folds)) {
    # Determine the cutoff index for the training set in this fold
    cutoff_train_end <- n - test_window - (n_folds - i) * test_window
    if (cutoff_train_end < 6) next # Skip this fold if there are too few training points
    
    # Define train and test indices based on the cutoff
    train_idx <- 1:cutoff_train_end
    test_idx  <- (cutoff_train_end + 1):(cutoff_train_end + test_window)
    
    # RAW- Extract raw training and testing data
    x_train_raw <- x_all[train_idx]; y_train_raw <- y_all[train_idx]
    x_test_raw  <- x_all[test_idx];  y_test_raw  <- y_all[test_idx]
    
    #  Scale features and targets based on training data only
    eps <- 1e-8
    x_min_t <- min(x_train_raw); x_max_t <- max(x_train_raw)
    y_min_t <- min(y_train_raw); y_max_t <- max(y_train_raw)
    denom_x <- max(eps, x_max_t - x_min_t)
    denom_y <- max(eps, y_max_t - y_min_t)
    
    x_train <- (x_train_raw - x_min_t) / denom_x
    y_train <- (y_train_raw - y_min_t) / denom_y
    x_test  <- (x_test_raw  - x_min_t) / denom_x
    
    # # Temporal validation split (last 'val_frac' portion of training data is validation set)
    val_size <- max(1, floor(length(x_train) * val_frac))
    val_idx <- (length(x_train) - val_size + 1):length(x_train)
    train_in_idx <- 1:(length(x_train) - val_size)
    
    # Separate into inner training and validation sets
    x_train_in <- x_train[train_in_idx]; y_train_in <- y_train[train_in_idx]
    x_val <- x_train[val_idx]; y_val <- y_train[val_idx]
    
    ## Clear previous Keras session to free memory
    keras::k_clear_session()
    # Build the model based on hyperparameters
    model <- keras_model_sequential()
    model %>% layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun, input_shape = 1) %>%
      layer_dropout(rate = hp_row$drop_rate)
    
    if (hp_row$layers >= 2) model %>% layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun)
    if (hp_row$layers >= 3) model %>% layer_dropout(rate = hp_row$drop_rate) %>% 
      layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun)
    if (hp_row$layers == 4) model %>% layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun)
    
    model %>% layer_dense(units = 1)
    
    # Compile the model with Mean Squared Error loss and the optimizer from hyperparameters
    model %>% compile(loss = "mse", optimizer = hp_row$opt_fun, metrics = list("mae"))
    
    # Early stopping to prevent overfitting and restore best weights
    cb <- callback_early_stopping(monitor = "val_loss", patience = 25, restore_best_weights = TRUE)
    
    # Train the model
    history <- model %>% fit(
      x = matrix(x_train_in, ncol = 1), y = matrix(y_train_in, ncol = 1),
      validation_data = list(matrix(x_val, ncol = 1), matrix(y_val, ncol = 1)),
      epochs = 200, batch_size = 16, callbacks = list(cb), verbose = 0
    )
    
    # Predict on the test set (scaled), then transform back to original scale
    pred_test_scaled <- model %>% predict(matrix(x_test, ncol = 1))
    pred_test <- as.numeric(pred_test_scaled) * denom_y + y_min_t
    
    # Store error metrics for this fold
    maes <- c(maes, mean(abs(pred_test - y_test_raw)))
    rmses <- c(rmses, sqrt(mean((pred_test - y_test_raw)^2)))
    # Clear session and remove model to free memory
    keras::k_clear_session(); gc()
    rm(model)
  }
  
  # Summarize fold-level metrics into a single tibble for the current hyperparameter set
  tibble(
    mae_fold_mean = mean(maes, na.rm = TRUE),   # Mean MAE across folds
    mae_fold_sd   = sd(maes, na.rm = TRUE),     # Standard deviation of MAE across folds
    rmse_fold_mean = mean(rmses, na.rm = TRUE), # Mean RMSE across folds
    rmse_fold_sd   = sd(rmses, na.rm = TRUE),   # Standard deviation of RMSE across folds
    n_folds_used = sum(!is.na(maes))            # Number of folds successfully evaluated
  )
}

# -------------------------------------------
# Main loop to run evaluation for all stocks
# -------------------------------------------
results_list <- list()
# Iterate over each unique stock in the dataset
for (stk in unique(ct$stock)) {
  ct_sub <- ct %>% filter(stock == stk) %>% arrange(year)
  x_all <- as.numeric(ct_sub$year)
  y_all <- as.numeric(ct_sub$ct)
  
  # Evaluate all hyperparameter configurations in 'trials' for this stock
  res <- foreach(h = seq_len(nrow(trials)), .combine = bind_rows,
                 .packages = c("keras","dplyr","tibble")) %dopar% {
                   hp_row <- trials[h, ]
                   eval_res <- eval_hp(hp_row, x_all, y_all,
                                       n_folds = n_folds,
                                       test_window = test_window)
                   tibble(stock = stk) %>%
                     bind_cols(hp_row) %>%
                     bind_cols(eval_res)
                 }
  # Store the results for this stock
  results_list[[stk]] <- res
}

stopCluster(cl)
# Combine results from all stocks into one final dataframe
final_results <- bind_rows(results_list)




#---------------------------------------------------------
#Selecting the best model and projecting data (2016-2025)
#---------------------------------------------------------

n_reps <- 5 # Number of repetitions for prediction per stock
years_pred <- 2016:2025

# Select the best hyperparameters per stock (lowest MAE)
top_hp_df <- final_results %>%
  filter(!is.na(mae_fold_mean)) %>%
  group_by(stock) %>%
  arrange(mae_fold_mean) %>%
  slice_head(n = 1) %>%
  ungroup()

# ---------------------------------------------------------
# Function to build a Keras model based on hyperparameters
# ---------------------------------------------------------
build_model_from_hp <- function(hp_row) {
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun, input_shape = 1) %>%
    layer_dropout(rate = hp_row$drop_rate)
  
  # Optional additional layers depending on 'layers' parameter
  if (hp_row$layers >= 2) model %>% layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun)
  if (hp_row$layers >= 3) model %>% layer_dropout(rate = hp_row$drop_rate) %>%
    layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun)
  if (hp_row$layers == 4) model %>% layer_dense(units = hp_row$unit_layer, activation = hp_row$act_fun)
  
  model %>% layer_dense(units = 1)
  model %>% compile(loss = "mse", optimizer = hp_row$opt_fun, metrics = list("mae"))
  model
}

# --------------------------------------------------
# Function to process predictions for a single stock
# --------------------------------------------------
process_stock <- function(stk) {
  message("Processing stock: ", stk)
  
  # Extract data for the given stock
  ct_sub_all <- ct %>% filter(stock == stk) %>% arrange(year)
  train_df <- ct_sub_all %>% filter(year <= 2015)
  if (nrow(train_df) < 6) {
    warning("Estoque ", stk, " tem poucos pontos (<6) — pulando.")
    return(NULL)
  }
  
  # Raw training data
  x_train_raw <- as.numeric(train_df$year)
  y_train_raw <- as.numeric(train_df$ct)
  
  # Normalize data (min-max scaling) using only training set
  eps <- 1e-8
  x_min_t <- min(x_train_raw); x_max_t <- max(x_train_raw)
  y_min_t <- min(y_train_raw); y_max_t <- max(y_train_raw)
  denom_x <- max(eps, x_max_t - x_min_t)
  denom_y <- max(eps, y_max_t - y_min_t)
  
  x_train <- (x_train_raw - x_min_t) / denom_x
  y_train <- (y_train_raw - y_min_t) / denom_y
  
  # Prepare prediction inputs (scaled)
  x_pred_raw <- years_pred
  x_pred_scaled <- (x_pred_raw - x_min_t) / denom_x
  
  # Retrieve best hyperparameters for this stock
  hp_row <- top_hp_df %>% filter(stock == stk) %>% slice(1)
  
  # Store predictions for each repetition
  preds_all <- matrix(NA_real_, nrow = n_reps, ncol = length(years_pred))
  
  # ---------------------------------------------------
  # Repeat model training and prediction 'n_reps' times
  # ---------------------------------------------------
  for (rep in seq_len(n_reps)) {
    set.seed(NULL)
    keras::k_clear_session()
    
    model <- build_model_from_hp(hp_row)
    
    #callback_es <- callback_early_stopping(monitor = "loss", patience = 30, restore_best_weights = TRUE)
    # Fit the model
    model %>% fit(
      x = matrix(x_train, ncol = 1), y = matrix(y_train, ncol = 1),
      epochs = 150,
      batch_size = 16,
      verbose = 0
      #callbacks = list(callback_es)
    )
    # Add noise to prediction inputs (for variability)
    x_pred_scaled_noisy <- pmax(
      x_pred_scaled * (1 + rnorm(length(x_pred_scaled), 0, 0.05)),
      0
    )
    pred_scaled <- as.numeric(model %>% predict(
      matrix(x_pred_scaled_noisy, ncol = 1),
      training = TRUE
    ))
    # Convert back to original scale and avoid negative predictions
    pred_unscaled <- pmax(pred_scaled * denom_y + y_min_t, 0)
    preds_all[rep, ] <- pred_unscaled
    
    # Clear model from memory
    keras::k_clear_session()
    gc()
    rm(model)
  }
  # Aggregate predictions: mean and 95% prediction interval
  mean_pred <- apply(preds_all, 2, mean, na.rm = TRUE)
  lower_pred <- apply(preds_all, 2, quantile, probs = 0.025, na.rm = TRUE)
  upper_pred <- apply(preds_all, 2, quantile, probs = 0.975, na.rm = TRUE)
  
  # Return as tibble
  tibble(
    stock = stk,
    year = years_pred,
    pred_mean = mean_pred,
    pred_lower_2.5 = lower_pred,
    pred_upper_97.5 = upper_pred
  )
}

# -------------------------------------------
# Run predictions for all stocks in parallel
# -------------------------------------------
plan(multisession, workers = parallel::detectCores())

# Run process_stock() for each stock in parallel
results_list <- future_map(unique(top_hp_df$stock), process_stock, .progress = TRUE)

# Remove NULL entries (stocks that were skipped)
results_list <- purrr::compact(results_list)

# Combine all results into a single dataframe
predictions_all <- bind_rows(results_list)

# Save results to CSV
readr::write_csv(predictions_all, "shrimp_predictions_top1hp_5reps_2016_2025.csv")
message("Saved predictions to shrimp_predictions_top1hp_5reps_2016_2025.csv")


# ------------------------
# Create plot_data_obs
# ------------------------
plot_data_obs <- ct %>%
  filter(year <= 2015) %>%         # Keep only observed data (up to 2015)
  select(stock, year, ct)          # Keep only necessary columns

# ------------------------
# Create plot_data_pred
# ------------------------
plot_data_pred <- predictions_all %>%
  dplyr::rename(ct = pred_mean)    # Rename "pred_mean" to "ct" to match observed data naming

# ---------------------------------------------------------------
# Separate stock information into components and mark as observed
# ---------------------------------------------------------------
plot_data_obs_sep <- plot_data_obs %>%
  separate(stock, into = c("category", "region", "source"), sep = "_") %>%  # Split stock into three columns
  mutate(
    type = "Observed",             # Mark as observed data
    pred_lower_2.5 = NA_real_,      # No prediction intervals for observed data
    pred_upper_97.5 = NA_real_
  )

# -----------------------------------------------------------------
# Separate stock information into components and mark as predicted
# -----------------------------------------------------------------
plot_data_pred_sep <- plot_data_pred %>%
  separate(stock, into = c("category", "region", "source"), sep = "_") %>%  # Split stock into three columns
  mutate(type = "Predicted")       # Mark as predicted data

# -----------------------------------------------------
# Combine observed and predicted datasets for plotting
# -----------------------------------------------------
plot_data_all <- bind_rows(plot_data_obs_sep, plot_data_pred_sep) %>%
  arrange(category, region, source, year) %>%                               # Sort by grouping variables
  mutate(source = factor(source, levels = c("Silva", "Freire")))            # Set factor order for source

# -------------
# Quick checks 
# -------------
print(table(plot_data_all$type))   # Check counts of observed vs predicted
print(unique(plot_data_all$source)) # Check unique data sources

# ---------------------------------------------------------------------
# Create plot: one continuous line across observed and predicted years
# --------------------------------------------------------------------
p2 <- ggplot(plot_data_all, 
             aes(x = year, y = ct / 1000, color = source, 
                 group = interaction(category, region, source))) +
  geom_line(linewidth = 1.5) +     # Continuous line (observed + predicted)
  #geom_point(data = filter(plot_data_all, type == "Observed"), size = 1.6) + # Uncomment to plot observed points
  geom_ribbon(
    data = filter(plot_data_all, type == "Predicted"),  # Only apply ribbon to predictions
    aes(
      x = year,
      ymin = pred_lower_2.5 / 1000, # Lower prediction interval
      ymax = pred_upper_97.5 / 1000, # Upper prediction interval
      fill = source,
      group = interaction(category, region, source)
    ),
    alpha = 0.25,                   # Transparency for prediction intervals
    inherit.aes = FALSE             # Do not inherit main plot aesthetics
  ) +
  geom_vline(xintercept = 2016, col = "gray60",linetype="dotted") +  # Vertical line marking start of predictions
  facet_wrap(region ~ category, scales = "free_y") + # Facet by region and category, free y-axis
  scale_colour_manual(values = c("Silva" = "gray60", "Freire" = "gray30")) + # Manual colors
  scale_fill_manual(values   = c("Silva" = "gray60", "Freire" = "gray30")) + # Manual fill colors
  labs(x = "Year", y = "Catch (1000 t)", color = "", fill = "") +            # Axis labels
  theme_classic(base_size = 11) %+replace%
  theme(
    strip.background = element_blank(), 
    plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"), # Tight plot margins
    axis.text.y =element_text(size = 11),
    axis.text.x =element_text(size = 11),
    strip.text.x = element_text(
      margin = margin(b = 1),  # Bottom margin for facet labels
      size = 11
    ),
    legend.box.margin = margin(t = -10), # Move legend closer
    legend.text = element_text(size=11),
    legend.spacing.y = unit(0.1, "cm"),  # Reduce spacing between legend rows
    legend.position = "bottom"           # Place legend at bottom
  )

# Display the plot
p2

# --------------------------
# Save plot as PNG
# --------------------------
ggsave("Catch_series.png", plot = p2, device = "png",  units = "cm", width = 35, height = 18)


# Select the best model (lowest mean fold) for each stock
best_models <- final_results %>%
  group_by(stock) %>%
  dplyr::arrange(mae_fold_mean, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  tidyr::separate(stock, c("category","region","source"), "_") %>%
  mutate(mae_fold_mean=round(mae_fold_mean, 1))

# Save as CSV
write.csv(best_models, "best_models_per_stock.csv", row.names = FALSE)
#----------------------------------------------------------------------------




# ------------------------------------------------------------------
# CMSY++ Neural Network function to estimate depletion B/k (end.bio)
# ------------------------------------------------------------------
# This function reproduces the CMSY++ logic to form predictors and
# feeds them into the trained neural net "nn.endbio" to obtain
# a *range* for end-of-series depletion B/K (lower, upper).
#
# Requirements:
# - An object `nn.endbio` in the workspace (loaded from ffnn.bin).
# - Normalization bounds in the workspace:
#     slope.first.min, slope.first.max
#     slope.last.min,  slope.last.max
#     (optional but recommended) yr.norm.min, yr.norm.max
#
# Notes:
# - All predictor column names must match exactly what the net expects.
# - Catch units cancel in ratios; keep the same units throughout.
# - The result is a two-number vector: c(lower_BK, upper_BK).
# ---------------------------------------------------------------

# 1) Load the trained nets and constants produced by CMSY++
#    (ffnn.bin must be in your working directory or use a full path)
load("ffnn.bin")   # loads nn.endbio and normalization bounds into the workspace

estimate_endbio <- function(ct_raw, yr) {
  # ---- Safety checks ----#
  if (!exists("nn.endbio")) {
    stop("`nn.endbio` not found. Make sure you loaded ffnn.bin before calling estimate_endbio().")
  }
  
  need_vars <- c("slope.first.min", "slope.first.max",
                 "slope.last.min",  "slope.last.max")
  
  if (!all(need_vars %in% ls(envir = .GlobalEnv))) {
    stop("Missing slope normalization bounds in workspace: ",
         paste(setdiff(need_vars, ls(envir = .GlobalEnv)), collapse = ", "))
  }
  
  
  # ---- Basic sizes  ----
  nyr <- length(yr)         # number of years in the series
  if (nyr < 5) {
    stop("Catch series is too short (<5 years). The neural net predictors cannot be formed reliably.")
  }
  
  # --------- MSY.pr (MSY prior) ---------
  # CMSY++ rule:
  # - If the maximum catch occurs in the last ~5 years OR series is very flat
  #   (low CV and min/max not too small), use the mean of the top-3 catches.
  # - Otherwise, use 0.75 * mean of the top-5 catches.
  ct.sort <- sort(ct_raw)
  sd.ct   <- sd(ct_raw)
  mean.ct <- mean(ct_raw)
  min.ct  <- min(ct_raw)
  max.ct  <- max(ct_raw)
  min_max <- min.ct / max.ct          # unitless: how far min is from max
  max.yr.i <- which.max(ct_raw)       # index of max catch (1..nyr)
  
  if (max.yr.i > (nyr - 4) || ((sd.ct / mean.ct) < 0.1 && min_max > 0.66)) {
    MSY.pr <- mean(ct.sort[(nyr-2):nyr])       # mean of top-3
  } else {
    MSY.pr <- 0.75 * mean(ct.sort[(nyr-4):nyr])# 0.75 * mean of top-5
  }
  
  # --------- int.yr (intermediate year) ---------
  # CMSY++ rule:
  # - If the time series is near-flat (min_max>0.7), use the mid year.
  # - Else, look within the last 30 years but 4 years away from endpoints,
  #   compare min vs max positions to pick an "intermediate" signal year.
  if (min_max > 0.7) {
    int.yr <- as.integer(mean(c(min(yr), max(yr))))
  } else {
    # restrict candidate years away from endpoints and within last 30 years
    yrs.int <- yr[yr > (yr[nyr] - 30) & yr > yr[4] & yr < yr[nyr - 4]]
    ct.int  <- ct_raw[yr > (yr[nyr] - 30) & yr > yr[4] & yr < yr[nyr - 4]]
    
    # fallback if the filter above yields nothing (very short series)
    if (length(yrs.int) == 0L) {
      int.yr <- as.integer(mean(c(min(yr), max(yr))))
    } else {
      min.ct.int    <- min(ct.int)
      min.ct.int.yr <- yrs.int[which.min(ct.int)]
      max.ct.int    <- max(ct.int)
      max.ct.int.yr <- yrs.int[which.max(ct.int)]
      
      if (min.ct.int.yr > max.ct.int.yr) {
        # if the min happens after the max, pick the min year
        int.yr <- min.ct.int.yr
      } else {
        # otherwise, see if the minimum after the max is sufficiently small
        min.ct.after.max <- min(ct.int[yrs.int >= max.ct.int.yr])
        if ((min.ct.after.max / max.ct.int) < 0.75) {
          int.yr <- yrs.int[yrs.int > max.ct.int.yr & ct.int == min.ct.after.max]
        } else {
          int.yr <- min.ct.int.yr
        }
      }
    }
  }
  
  # value of ct/MSY.pr for the chosen intermediate year
  ct_MSY.int <- ct_raw[yr == int.yr] / MSY.pr
  
  # --------- Indices and normalized counts ---------
  # Positions (as fractions of series length) and normalized length:
  min.ct.i <- which.min(ct_raw) / nyr
  max.ct.i <- which.max(ct_raw) / nyr
  int.ct.i <- which(yr == int.yr) / nyr
  
  # Year-count normalization:
  # If CMSY++ global bounds for yr.norm exist, use them (preferred).
  # Otherwise fallback to simple scaling to [0,1] from local min/max year.
  if (all(c("yr.norm.min","yr.norm.max") %in% ls())) {
    yr.norm <- (nyr - get("yr.norm.min")) / (get("yr.norm.max") - get("yr.norm.min"))
  } else {
    yr.norm <- (nyr - min(yr)) / (max(yr) - min(yr))
  }
  
  # --------- Catch means near start and end (scaled by MSY.pr) ---------
  # mean of first 5 years / MSY.pr; mean of last 5 years / MSY.pr
  # (guard if series shorter than 10 years)
  k_start <- min(5, nyr)                   # up to first 5 points
  k_end   <- min(5, nyr)                   # up to last 5 points
  mean.ct_MSY.start <- mean(ct_raw[1:k_start]) / MSY.pr
  mean.ct_MSY.end   <- mean(ct_raw[(nyr-k_end+1):nyr]) / MSY.pr
  
  # --------- Slopes (first & last ~10 years), normalized ---------
  # Slopes are computed on catch standardized by its mean.
  # If the series is shorter than 10 points, use the available length.
  m_first <- min(10, nyr)
  m_last  <- min(10, nyr)
  slope.first <- coef(lm( (ct_raw[1:m_first] / mean.ct) ~ seq_len(m_first) ))[2]
  slope.last  <- coef(lm( (ct_raw[(nyr-m_last+1):nyr] / mean.ct) ~ seq_len(m_last) ))[2]
  
  # Normalize slopes to [0,1] using the global min/max from CMSY++ training
  slope.first.nrm <- (slope.first - get("slope.first.min")) /
    (get("slope.first.max") - get("slope.first.min"))
  slope.last.nrm  <- (slope.last  - get("slope.last.min"))  /
    (get("slope.last.max")  - get("slope.last.min"))
  
  # --------- Catch-shape flags (Flat, LH, LHL, HL, HLH, else OTH) ---------
  # These are binary indicators characterizing the time-series shape,
  # using min/max ratio and relative catches at start and end.
  min_max <- min(ct_raw) / max(ct_raw)
  start.rel <- ct_raw[1]   / max(ct_raw)
  end.rel   <- ct_raw[nyr] / max(ct_raw)
  
  Flat <- as.numeric(min_max >= 0.45 && start.rel >= 0.45 && end.rel >= 0.45)
  LH   <- as.numeric(min_max <  0.25 && start.rel <  0.45 && end.rel >  0.45)
  LHL  <- as.numeric(min_max <  0.25 && start.rel <  0.45 && end.rel <  0.25)
  HL   <- as.numeric(min_max <  0.25 && start.rel >  0.50 && end.rel <  0.25)
  HLH  <- as.numeric(min_max <  0.25 && start.rel >= 0.45 && end.rel >= 0.45)
  OTH  <- as.numeric(sum(c(Flat, LH, LHL, HL, HLH)) < 1)  # "other" if none matched
  
  # --------- Build predictor frame with EXACT column names ---------
  # IMPORTANT: These names must match the net's training variables.
  preds <- data.frame(
    Flat, LH, LHL, HL, HLH, OTH,
    ct_MSY.int,
    min_max, max.ct.i, int.ct.i,
    min.ct.i, yr.norm,
    mean.ct_MSY.start, slope.first.nrm,
    mean.ct_MSY.end,   slope.last.nrm
  )
  
  # --------- Neural net inference ---------
  # The network outputs two-class logits; we select the class
  # (1 = B/K < 0.5, 2 = B/K >= 0.5), then map to a B/K interval.
  pr.nn <- neuralnet::compute(nn.endbio, preds)
  idx   <- max.col(pr.nn$net.result)  # 1 or 2 (class index)
  
  # use min of last-year vs last-5-years ratio to be conservative (CMSY++ rule)
  ct_MSY.end <- ct_raw[nyr] / MSY.pr
  ct_MSY.use <- min(ct_MSY.end, mean.ct_MSY.end)
  
  # --------- CMSY++ linear mapping to B/K bounds ---------
  # If mean(ct/MSY.pr) over last 5 years >= 1, use the broad bk.MSY range.
  # Else, choose the "below 0.5" or "above 0.5" linear band by class `idx`.
  bk.MSY   <- c(0.256, 0.721)  # generic 90% range if ct/MSY.pr >= 1
  CL.1     <- c(0.01,  0.203)  # below-0.5 lower bound: a + b * x
  CL.2     <- c(0.20,  0.431)  # below-0.5 upper bound: a + b * x
  CL.3     <- c(0.80, -0.45 )  # above-0.5 lower bound: a + b * x
  CL.4     <- c(1.02, -0.247)  # above-0.5 upper bound: a + b * x
  ct_MSY.lim <- 1.0
  
  if (mean.ct_MSY.end >= ct_MSY.lim) {
    return(bk.MSY)
  } else if (idx == 1) {
    # class 1 => B/K likely < 0.5
    return(c(CL.1[1] + CL.1[2] * ct_MSY.use,
             CL.2[1] + CL.2[2] * ct_MSY.use))
  } else {
    # class 2 => B/K likely >= 0.5
    return(c(CL.3[1] + CL.3[2] * ct_MSY.use,
             CL.4[1] + CL.4[2] * ct_MSY.use))
  }
}

# Example:
ct_raw <- ct$ct[ct$stock == "brown_NE_Silva"]
yr     <- ct$year[ct$stock == "brown_NE_Silva"]

endbio <- estimate_endbio(ct_raw, yr)
endbio  # -> c(lower_BK, upper_BK)


#--------------------------------------------------------------------------------
#Organizing catches

# Filtering from 2016-2025
predictions_filtered <- predictions_all %>%
  filter(year >= 2016 & year <= 2025) %>%
  select(stock, year, pred_mean) %>%
  dplyr::rename(ct = pred_mean)

# Separate 'stock' in category , region and source
predictions_filtered <- predictions_filtered %>%
  separate(stock, into = c("category", "region", "src"), sep = "_", remove = FALSE) %>%
  mutate(source = "projection") %>%  # projection
  select(year, category, region, ct, source, stock)

# Bind rows with reconstructed data (Freire and Silva)
ct_full <- bind_rows(ct, predictions_filtered) %>%
        arrange(year)

# head..
head(ct_full)

#write data
# Save as CSV
write.csv(ct_full, "ct_projected_NN.csv", row.names = FALSE)
#--------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Apply the Artificial Neural Network (ANN, CMSY++) and the zBRT (Zhou) methods 
# to estimate biomass depletion (B/K) in 2015 and 2025.
# Modified by Silva MLS
# Generic Function that runs ANN and zBRT for a single stock
# ------------------------------------------------------------------------------
run_bk_methods <- function(s, data) {
  sub <- data %>% filter(stock == s)   # subset data for the selected stock
  
  year  <- sub$year
  catch <- sub$ct
  
  results <- list()
  
  ## -------------------------
  ## Neural Network (CMSY++)
  ## -------------------------
  for (yr_target in c(2015, 2025)) {
    try({
      # Restrict time series up to the target year
      ct_raw <- catch[year <= yr_target]
      yr_sub <- year[year <= yr_target]
      
      # Apply neural net to estimate lower and upper B/K bounds
      bk_nn <- estimate_endbio(ct_raw, yr_sub)
      
      # Take the midpoint between lower and upper bound
      bk_mid <- mean(bk_nn)
      
      # Save results for this stock and year
      results[[length(results) + 1]] <- data.frame(
        stock   = s,
        year    = yr_target,
        method  = "NN",
        bk_lo   = bk_nn[1],
        bk_hi   = bk_nn[2],
        bk      = bk_mid
      )
    }, silent = TRUE)
  }
  
  ## -------------------------
  ## zBRT
  ## -------------------------
  try({
    # Run zBRT using the full catch time series
    output <- zbrt(year, catch)
    
    for (yr_target in c(2015, 2025)) {
      # Extract only if the target year is available in the zBRT output
      if (yr_target %in% output$ts$year) {
        idx <- which(output$ts$year == yr_target)
        
        results[[length(results) + 1]] <- data.frame(
          stock   = s,
          year    = yr_target,
          method  = "BRT",
          bk_lo   = output$ts$s_lo[idx],
          bk_hi   = output$ts$s_hi[idx],
          bk      = output$ts$s[idx]
        )
      }
    }
  }, silent = TRUE)
  
  # Return combined results as a data.frame
  if (length(results) > 0) {
    return(bind_rows(results))
  } else {
    return(NULL)
  }
}

# --------------------------
# Parallel loop setup
# --------------------------
# Detect number of cores and create a cluster
n_cores <- parallel::detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Export custom functions to the cluster workers
clusterExport(cl, varlist = c("estimate_endbio", "zbrt", "run_bk_methods"))

# Each worker must load required packages and the neural net model (ffnn.bin)
clusterEvalQ(cl, {
  library(dplyr)
  library(tidyr)
  library(neuralnet)
  load("ffnn.bin")   ## loads nn.endbio and bounds for ANN
  library(datalimited2)
  NULL
})

# -------------------------------
# Run in parallel for all stocks
# -------------------------------
stocks <- unique(ct_full$stock)

bk_all_list <- foreach(s = stocks, .packages = c("dplyr", "tidyr", "neuralnet")) %dopar% {
  message("Running stock: ", s)  # print on console
  tryCatch({
    run_bk_methods(s, data = ct_full)
  }, error = function(e) {
    message(paste("Error in stock:", s, ":", e$message))
    NULL
  })
}

# Stop the cluster and revert to sequential execution
stopCluster(cl)
registerDoSEQ()

# ----------------
# Combine results
# ----------------
bk_all <- bind_rows(bk_all_list) %>%
  separate(stock,
           into = c("category", "region", "source"),
           sep = "_",
           remove = FALSE) %>%
  # calcula flags primeiro
  mutate(
    collapsed = (bk - 0.1) < 0.2,
    low_bk_tropical = region %in% c("N", "NE") & bk < 0.5
  ) %>%
  # inicia bk_mean_adj com o valor original bk
  mutate(bk_mean_adj = bk)

# agora preenche por máscara, gerando uma rnorm por linha onde for necessário
# 1) estoques tropicais (N/NE) com bk < 0.5 → sample ~ Normal(mean=0.5, sd=0.1)
mask_tropical <- bk_all$low_bk_tropical
if (any(mask_tropical, na.rm = TRUE)) {
  bk_all$bk_mean_adj[mask_tropical] <- rnorm(sum(mask_tropical, na.rm = TRUE),
                                             mean = 0.5, sd = 0.02)
}

# 2) estoques "colapsados" (aplica apenas onde não já foi tratado como tropical)
mask_collapsed <- bk_all$collapsed & !mask_tropical
if (any(mask_collapsed, na.rm = TRUE)) {
  bk_all$bk_mean_adj[mask_collapsed] <- rnorm(sum(mask_collapsed, na.rm = TRUE),
                                              mean = 0.3, sd = 0.02)
}

# garante limites e calcula intervalos
bk_all <- bk_all %>%
  mutate(
    bk_mean_adj = pmin(pmax(bk_mean_adj, 0.2), 0.8),
    bk_lo = pmax(bk_mean_adj - 0.1, 0.0),
    bk_hi = pmin(bk_mean_adj + 0.1, 1.0),
    bk = bk_mean_adj,
    stock_ind = paste(category, region, sep = "_")
  ) %>%
  select(-collapsed, -low_bk_tropical) %>%
  #extra filter
    mutate(
    bk = if_else(region %in% c("N", "NE") & bk < 0.5, 0.5, bk),
    bk_lo = if_else(region %in% c("N", "NE") & bk < 0.5, 0.4, bk_lo),
    bk_hi = if_else(region %in% c("N", "NE") & bk < 0.5, 0.6, bk_hi)
  )





#plot depletions Bk
p3 <- ggplot(filter(bk_all, year == 2015), aes(
  x = method, y = bk,
  ymin = bk_lo, ymax = bk_hi,
  color = method,
  shape = stock_ind,
  group = interaction(category, region, method, source)
)) +
  geom_linerange(position = position_dodge(width = 0.85), linewidth = 1.2) +
  geom_point(position = position_dodge(width = 0.85), size = 3.5,stroke = 1.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  scale_shape_manual(values = 0:11) +
  scale_colour_viridis_d()+
  scale_y_continuous(breaks = seq(0.1,0.8,0.1))+
  labs(
    x = "Estimation method", 
    y =  expression("Biomass depletion (B/k"[2015]*")"), 
    color = "",
    shape = ""
  ) +
  theme_bw(base_size = 15) +
  theme(
    legend.position = "bottom",
    plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"),
    strip.text.x = element_text(margin = margin(b = 1), size = 11),
    axis.text.y = element_text(size = 15),
    axis.text.x = element_text(size = 15, face = "italic"),
    legend.text = element_text(size=14),
    legend.box.margin = margin(t = -10),
    legend.spacing.y = unit(0.1, "cm")
  )

p3


#save png..
ggsave("bk_priors.png", plot = p3, device = "png",  units = "cm", width = 30, height = 17)


#----------------------------------------------------------------------------------------




#--------------------------------------------------------------------------
# Create prior rmax based on life-history traits (Cortes, 2016)
# Modified by Silva mls
# Parametric bootstrap to propagate uncertainty
# Import life history values
#--------------------------------------------------------------------------

# Safe version of uniroot: tries to expand the search interval
# If no sign change is found or errors occur, returns NA instead of breaking
safe_uniroot <- function(fn, lower = 0, upper = 5, tol = 1e-8, max_expand = 10, by = 0.5) {
  safe_eval <- function(x) tryCatch(fn(x), error = function(e) NA_real_)
  f_low <- safe_eval(lower)
  f_high <- safe_eval(upper)
  # If valid and sign change exists, run uniroot
  if (!is.na(f_low) && !is.na(f_high) && f_low * f_high < 0) {
    return(tryCatch(uniroot(fn, c(lower, upper), tol = tol)$root,
                    error = function(e) NA_real_))
  }
  # Otherwise expand the interval progressively
  for (i in seq_len(max_expand)) {
    new_upper <- upper + i * by
    f_new <- safe_eval(new_upper)
    if (!is.na(f_low) && !is.na(f_new) && f_low * f_new < 0) {
      return(tryCatch(uniroot(fn, c(lower, new_upper), tol = tol)$root,
                      error = function(e) NA_real_))
    }
  }
  # If nothing works, return NA
  return(NA_real_)
}

# Helper: generate lognormal samples from mean and CV
# Ensures positive values for life-history parameters
rlnorm_from_mean_cv <- function(mean, cv, n) {
  if (is.na(mean) || is.na(cv) || mean <= 0) return(rep(NA_real_, n))
  sigma2 <- log(1 + cv^2)
  mu <- log(mean) - 0.5 * sigma2
  rlnorm(n, meanlog = mu, sdlog = sqrt(sigma2))
}

# Wrapper bootstrap for a single species
# Returns both all simulated values and summaries (median, quantiles, convergence)
estimate_r_boot <- function(sp_row, nboot = 1000,
                            cvs = list(Linf = 0.15, k = 0.20, M = 0.20, tmax = 0.10, L50 = 0.15, ls = 0.25),
                            ls_euler_fixed = 9.5,
                            options = list(r_upper = 5, verbose = FALSE)) {
  sp <- sp_row$Especie[1]
  
  # Mean point estimates from data
  Linf0 <- mean(na.omit(sp_row$`Linf(mm)TL`)) / 10
  k0    <- mean(na.omit(sp_row$`K(ano)`))
  L500  <- mean(na.omit(sp_row$`L50(mm)TL`)) / 10
  M0    <- mean(na.omit(sp_row$M))
  tmax0 <- mean(na.omit(sp_row$`Tmáx`))
  ls0   <- if("ls" %in% names(sp_row)) mean(na.omit(sp_row$ls)) else 4
  f     <- if("f" %in% names(sp_row)) mean(na.omit(sp_row$f)) else 2
  
  # If missing parameters, return NA tables
  if (any(is.na(c(Linf0, k0, L500, M0, tmax0)))) {
    if (isTRUE(options$verbose)) warning(sp, ": insufficient parameters.")
    return(list(
      sims = tibble(),
      summary = tibble(specie = sp, method = c("euler","myers","smith_rebound_eq6","demographic_inv"),
                       r_median = NA_real_, r_q025 = NA_real_, r_q975 = NA_real_,
                       n_conv = 0L, n_total = nboot)
    ))
  }
  
  # Generate parametric bootstrap samples
  Linf_samps <- rlnorm_from_mean_cv(Linf0, cvs$Linf, nboot)
  k_samps    <- rlnorm_from_mean_cv(k0, cvs$k, nboot)
  M_samps    <- rlnorm_from_mean_cv(M0, cvs$M, nboot)
  tmax_samps <- pmax(1, round(rlnorm_from_mean_cv(tmax0, cvs$tmax, nboot)))
  L50_samps  <- rlnorm_from_mean_cv(L500, cvs$L50, nboot)
  ls_samps   <- rlnorm_from_mean_cv(ls0, cvs$ls, nboot)
  
  # Internal function: run the four estimators of r for one parameter set
  run_one <- function(Linf, k, L50, M, tmax, ls) {
    if (is.na(Linf) || Linf <= 0 || is.na(k) || is.na(L50) || is.na(M) || is.na(tmax)) 
      return(c(NA,NA,NA,NA))
    if (L50 >= Linf) L50 <- 0.5 * Linf
    
    # Calculate age at 50% maturity, survivorship, fecundity
    t50 <- - (log(1 - L50 / Linf) / k)
    ages <- 0:ceiling(tmax)
    lx <- exp(-M * ages)
    mat_a <- 1 / (1 + exp(-(ages - t50)))
    fr <- ls / f / 2
    mx <- fr * mat_a
    fr_euler <- ls_euler_fixed / f / 2
    mx_euler <- fr_euler * mat_a
    
    # Define root-finding functions for each method
    euler_fn <- function(r) sum(lx * mx_euler * exp(-r * ages)) - 1
    s_adult <- lx[which.min(abs(lx - 0.5))]
    litter <- ls; freqv <- f; tmat <- t50
    formula_myers <- function(rm) ((exp(rm))^tmat) - ( (s_adult) * ((exp(rm))^(tmat - 1)) ) - (litter / freqv / 2)
    Z <- 1.5 * M
    l_alpha <- if ((tmax - tmat + 1) > 0) (1 - exp(-Z)) / ((litter/2/freqv) * (1 - exp(-Z*(tmax - tmat + 1)))) else NA
    eq6 <- function(reb) if (is.na(l_alpha)) NA_real_ else 
      1 - exp(-(M + reb)) - l_alpha * (litter/2/freqv) * 1.25 * exp(-reb*tmat) * (1 - exp(-(M+reb)*(tmax - tmat + 1)))
    formula5 <- function(r) if (exp(r) <= s_adult) NA_real_ else 
      exp(r) - (exp(1 / (tmat + 1 + (s_adult / (exp(r) - s_adult)))))
    
    # Apply safe_uniroot for each method
    up <- options$r_upper %||% 5
    r1 <- safe_uniroot(euler_fn, 0, up)
    r2 <- safe_uniroot(formula_myers, 0, up)
    r3 <- safe_uniroot(eq6, 0, up)
    r4 <- safe_uniroot(formula5, 0, up)
    return(c(r1, r2, r3, r4))
  }
  
  # Run all bootstraps with pmap_dfr (vectorized loop)
  sims <- purrr::pmap_dfr(list(Linf_samps, k_samps, L50_samps, M_samps, tmax_samps, ls_samps),
                          function(Linf, k, L50, M, tmax, ls) {
                            rvec <- run_one(Linf, k, L50, M, tmax, ls)
                            tibble(r_euler = rvec[1], r_myers = rvec[2], r_eq6 = rvec[3], r_f5 = rvec[4])
                          }) %>%
    dplyr::mutate(iter = dplyr::row_number(), specie = sp)
  
  # Summaries per method
  summarize_method <- function(x) {
    n_conv <- sum(!is.na(x))
    tibble(median = median(x, na.rm = TRUE),
           q025   = quantile(x, 0.025, na.rm = TRUE),
           q975   = quantile(x, 0.975, na.rm = TRUE),
           n_conv = n_conv)
  }
  
  s1 <- summarize_method(sims$r_euler)
  s2 <- summarize_method(sims$r_myers)
  s3 <- summarize_method(sims$r_eq6)
  s4 <- summarize_method(sims$r_f5)
  
  summary_tbl <- tibble(
    specie = sp,
    method = c("Euler","Myers","Smith rebound","Demographic inv"),
    r_median = c(s1$median, s2$median, s3$median, s4$median),
    r_q025   = c(s1$q025,   s2$q025,   s3$q025,   s4$q025),
    r_q975   = c(s1$q975,   s2$q975,   s3$q975,   s4$q975),
    n_conv   = c(s1$n_conv, s2$n_conv, s3$n_conv, s4$n_conv),
    n_total  = nboot
  )
  
  return(list(sims = sims, summary = summary_tbl))
}

# Apply to all species
species_list <- unique(lf$Especie)

res_list <- map(species_list, function(sp) {
  sp_row <- lf %>% filter(Especie == sp)
  estimate_r_boot(sp_row, nboot = 200)
})

# Combine all simulations across species
r_sims <- map_dfr(res_list, "sims")
write.csv(r_sims, "r_sims.csv", row.names = FALSE)

# Combine all summaries across species
r_summary <- map_dfr(res_list, "summary") %>%
  mutate(
    r_min = pmax(r_median - 0.2, 0.01),
    r_max = r_median + 0.2
  ) %>%
  mutate(across(where(is.numeric), ~ round(.,2)))

write.csv(r_summary, "r_summary.csv", row.names = FALSE)


# Reshape for the long format
all_sims_long <- r_sims %>%
  pivot_longer(cols = starts_with("r_"),
               names_to = "method", values_to = "r") %>%
  mutate(method = dplyr::recode(method,
                                r_euler = "Euler",
                                r_myers = "Myers",
                                r_eq6   = "Smith rebound",
                                r_f5    = "Demographic inv"))

# Violin plot (Intrinsic growth rate)
p4<- ggplot(all_sims_long, aes(x = specie, y = r,col=method, fill = method)) +
  geom_boxplot(aes(fill=method,col = method), alpha = 0.4,width = 0.3, position = position_dodge(width = 0.8))+
  geom_violin(aes(col = method),trim = TRUE, alpha = 0.5,width = 1.5, position = position_dodge(width = 0.8)) +
  geom_jitter(aes(col = method),
              position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8),
              size = 1, alpha = 0.3) +
  labs(x = "Species", y = "Intrinsic growth rate (r)",
       fill = "", color = "") +
  scale_y_continuous(limits = c(0,1.5), breaks = seq(0,1.5,0.1))+
  scale_color_viridis_d()+
  scale_fill_viridis_d()+
  theme_classic(base_size = 15) %+replace%
  theme(
    strip.background = element_blank(), 
    plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"),
    strip.text.x = element_text(margin = margin(b = 1), size = 15),
    axis.text.y = element_text(size=15),
    axis.text.x = element_text(size = 15,  face = "italic"),
    legend.text = element_text(size=15),
    legend.box.margin = margin(t = -10),
    legend.spacing.y = unit(0.1, "cm"),
    legend.position = "bottom"
  )
p4

#save png..
ggsave("r_priors.png", plot = p4, device = "png",  units = "cm", width = 32, height = 17)




#------------------------------------------------------------
# Creating input data frames for CMSY (Froese et al., 2023)
# Using two reconstructed catch series (Freire and Silva)
# Using BRT+Literature+CMSY NN for prior depletions
# Using Euler-Lotka, Myers.. for intrinsic growth rate
#------------------------------------------------------------

#first create the cdat data frame

#replicate catch history for each scenario and prior
stocks=unique(ct_full$stock)
scenario= c("Base") #only base case scenarios for now
bk_method= c("NN","BRT") 
r_method= unique(r_summary$method)

#Expand combinations (each catch series for each combination of scenario,bk_method,r_method) many-to-many
df <- crossing(
  stock = stocks,
  scenario = scenario,
  bk_method = bk_method,
  r_method = r_method
) %>%
  # combine with ct_full only until 2015 (bind the history catch data)
  left_join(
    ct_full %>% filter(year <= 2015, source != "projection"),
    by = "stock",
    relationship = "many-to-many"
  ) %>%
  mutate(stock_id = paste(stock, scenario, bk_method, r_method, sep = "_"))
  

# Ct data for base case scenarios (CMSY format)
ct_base <- df %>%
  transmute(
    Stock = stock_id,   # #stock names (scenario+bk_method+r_method)
    yr    = year,       #year vector
    ct    = ct,         #catches
    bt    = NA          #biomass index
  )

#Ct data for projection scenarios (CMSY format)
ct_proj<- data.frame(
  Stock=paste(ct_full$stock, "Projection",sep = "_"), #stock names
  yr=ct_full$year,  #year vector
  ct=ct_full$ct,   #catches
  bt=NA           #biomass index
) 

#Bind rows
cdat <- bind_rows(ct_base, ct_proj)


#------------------------------------------------------------
# Creating the cinfo data frame (metadata for CMSY runs)
# Each row corresponds to one stock-scenario combination
# Derived from the Stock IDs in cdat
#------------------------------------------------------------
cinfo <- cdat %>%
  # Ensure each stock ID appears only once
  distinct(Stock) %>%
  # Extract identifiers and metadata embedded in the Stock name
  mutate(
    # Extract scenario type (Base or Projection)
    scenario   = str_extract(Stock, "Base|Projection"),
    # Extract depletion estimation method (Neural Network or BRT)
    bk_method  = str_extract(Stock, "NN|BRT"),
    # Extract intrinsic growth rate estimation method
    r_method   = str_extract(Stock, "Euler|Myers|Smith rebound|Demographic inv"),
    # Fixed attributes describing geographical origin
    Continent  = "South America",
    # Extract region (North, Northeast, Southeast, South)
    Region     = str_extract(Stock, "(?<=_)(NE|SE|N|S)(?=_)"),
    Subregion  = Region,   # same value for simplicity
    # Broad taxonomic or fishery group
    Group      = "Shrimp",
    # Extract common species name
    Name       = str_extract(Stock, "brown|pink|seabob|white"),
  
    # Assign scientific names based on species and region
    ScientificName = case_when(
      Name == "brown" & Region %in% c("N", "NE") ~ "F. subtilis",
      Name == "pink"  & Region == "SE" ~ "F. brasiliensis",
      Name == "pink"  & Region == "S"  ~ "F. paulensis",
      Name == "white" ~ "L. schmitti",
      Name == "seabob" ~ "X. kroyeri",
      TRUE ~ NA_character_
    ),
    
    # Create a simplified species code (e.g., seabob_NE)
    SpecCode = paste(Name, Region, sep = "_"),
    # Extract the data source (Freire or Silva)
    Source   = str_extract(Stock, "Freire|Silva"),
    #------------------------------------------------------------
    # Temporal coverage of the catch data for each stock
    #------------------------------------------------------------
    MinOfYear = map_dbl(Stock, ~min(cdat$yr[cdat$Stock == .x])),
    MaxOfYear = map_dbl(Stock, ~max(cdat$yr[cdat$Stock == .x])),
    StartYear = MinOfYear,
    EndYear   = MaxOfYear,
    #------------------------------------------------------------
    # Reference management fields (not used in CMSY but kept for structure)
    #------------------------------------------------------------
    Flim=NA, Fpa=NA, Blim=NA, Bpa=NA, Bmsy=NA, MSYBtrigger=NA, Fmsy=NA, last_F=NA,
    #------------------------------------------------------------
    # Biological priors and resilience information
    #------------------------------------------------------------
    Resilience = "Medium",   # ignored if r limits are explicitly set below
    # Lower and upper bounds of r (only for Base scenarios)
    r.low = ifelse(scenario == "Base",
                   round(r_summary$r_min[match(paste(ScientificName, r_method), 
                                              paste(r_summary$specie, r_summary$method))], 2), NA),
    r.hi  = ifelse(scenario == "Base",
                   round(r_summary$r_max[match(paste(ScientificName, r_method), 
                                               paste(r_summary$specie, r_summary$method))], 2), NA),
    #--------------------------------------
    # Prior depletion levels (B/k ratios)
    #--------------------------------------
    # Starting biomass (assumed near virgin state)
    stb.low = 0.8,
    stb.hi  = 1,
    # Intermediate year for mid-period depletion
    int.yr = NA, #ifelse(scenario == "Base",
                 #   bk_all$year[match(paste(Name, Region, Source), 
                  #                    paste(bk_all$category, bk_all$region, bk_all$source))], NA),
    # Lower and upper bounds for intermediate depletion (B/k)
    intb.low = NA, #ifelse(scenario == "Base",
                   #   round(bk_all$bk_lo[match(paste(Name, Region, Source), 
                    #                           paste(bk_all$category, bk_all$region, bk_all$source))], 2), NA),
    intb.hi = NA, #ifelse(scenario == "Base",
                   #  round(bk_all$bk_hi[match(paste(Name, Region, Source), 
                    #                          paste(bk_all$category, bk_all$region, bk_all$source))], 2), NA),
    # Lower and upper bounds for final depletion (B/k in 2015)
    endb.low = ifelse(scenario == "Base",
                      round(bk_all$bk_lo[match(paste(Name, Region, Source, bk_method, 2015),
                                               paste(bk_all$category, bk_all$region, bk_all$source, bk_all$method, bk_all$year))], 2),NA),
    endb.hi = ifelse(scenario == "Base",
                     round(bk_all$bk_hi[match(paste(Name, Region, Source, bk_method, 2015),
                                              paste(bk_all$category, bk_all$region, bk_all$source, bk_all$method, bk_all$year))], 2),NA),
    #------------------------------------------------------------
    # Additional CMSY input parameters
    #------------------------------------------------------------
    btype = "None",        # no external biomass time series
    e.creep = NA,          # no effort creep adjustment
    force.cmsy = TRUE,     # ensures CMSY accepts input even without bt
    Comments = NA          # optional notes placeholder
  )


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#write input files for CMSY++ (cdat and cinfo)
file1  <- "cdat_shrimp.csv"
write.table(cdat, file = file1, append =FALSE,dec=".",sep = ",",
            row.names = FALSE) 
file2  <- "cinfo_shrimp.csv"
write.table(cinfo, file = file2, append =FALSE,dec=".",sep = ",",
            row.names = FALSE) 
#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X#X#x#X#X

                              



                    #x#X#X#x#X#X#x#X#X#x#X#x#X#X#x#X#X#x#X#X#x#X#x#X#X#x#X#X#x#X#X#x#X
                    #                   Shrimp Stock Assessment                      #
                    #                   CMSY++ 16 (January 2021)                     #
                    #x#X#X#x#X#X#x#X#X#x#X#x#X#X#x#X#X#x#X#X#x#X#x#X#X#x#X#X#x#X#X#x#X  



##---------------------------------------------------------------------------------------------
## CMSY and BSM analysis ----
## Developed by Rainer Froese, Gianpaolo Coro and Henning Winker in 2016, version of January 2021
## PDF creation added by Gordon Tsui and Gianpaolo Coro
## Time series within 1950-2030 are stored in csv file
## Correction for effort creep added by RF
## Multivariate normal r-k priors added to CMSY by HW, RF and GP in October 2019
## Multivariate normal plus observation error on catch added to BSM by HW in November 2019
## Retrospective analysis added by GP in November 2019
## Bayesian implementation of CMSY added by RF and HW in May 2020
## Slight improvements to NA rules for prior B/k done by RF in June 2020
## RF added on-screen proposal to set start.year to medium catch if high or low biomass is unclear at low catch
## Alling notation and posterior compuations between CMSY++ and BSM done by HW in June 2020
## RF fixed a bug where some CMSY instead of BSM results were wrongly reported for management, October 2020
## RF updated cor.log.rk to -0.76 based and MSY.prior based om max.ct, based on a analysis of 240+ global stocks
## HW added use of MSY.prior to predict k.prior in JAGS
## RF and GP reviewed and improved B/k default priors, adding neural network
## HW added beta distribution for B/k priors
## GP added ellipse estimation (lower right focus) of most likely r-k pair for CMSY
##---------------------------------------------------------------------------------------------

# Automatic installation of missing packages
list.of.packages <- c("R2jags","coda","parallel","foreach","doParallel","gplots","mvtnorm","neuralnet","conicfit")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
library(devtools)
library(datalimited2)
library(mgcv)
library(dplyr)
library(tidyr)
library(plyr)
library(tibble)
library(keras)
library(furrr)
library(future)
library(purrr)
library(readr)
library(ggplot2)
library(R2jags)  # Interface with JAGS (download also: https://sourceforge.net/projects/mcmc-jags/)
library(coda)
library(gplots)
library(mvtnorm)
#library(snpar)
library(neuralnet)
library(conicfit)
library(geobr)
library(sf)
library(rnaturalearth)
library(caret)
library(foreach)
library(doParallel)
library(rlang)   # Helpers (e.g., %||%)
library(stringr)
library(patchwork)
library(stringr)
library(scales)


#defining workspace......
#rm(list = ls())
graphics.off()
#dir="C:/Matheus/Universidade/Doutorado/Avaliação Shrimp_CMSY"
#setwd(dir)

library(R2jags)  # Interface with JAGS
library(coda)
library(gplots)
library(mvtnorm)
#library(snpar)
library(neuralnet)
library(conicfit)
#-----------------------------------------
# Some general settings ----
#-----------------------------------------
# set.seed(999) # use for comparing results between runs
#rm(list=ls(all=FALSE)) # clear previous variables etc
options(digits=3) # displays all numbers with three significant digits as default
graphics.off() # close graphics windows from previous sessions
FullSchaefer <- F    # initialize variable; automatically set to TRUE if enough abundance data are available
n.chains     <- 2 # number of chains to be used in JAGS, default = 2
#setwd("C:/Matheus/Universidade/Doutorado/Avaliação Shrimp_CMSY") # set working directory to source file location

#-----------------------------------------
# Required settings, File names ----
#-----------------------------------------
catch_file  <- "cdat_shrimp.csv" #"Stocks_Catch_2020CMSYrun2_v5_RS - Copy.csv"  #"CombStocks_Catch_2020CMSYrun3_v4.csv"  # "SAUP_Catch_1.csv"  #"SimCatchCPUE_4.csv"  # "Stocks_Catch_Aust_2.csv" #"STECF_Catch_2020_2.csv" #"tRFMO_Catch_2020.csv" #"ICES_Catch_2020.csv" #"Global_Stocks_Catch.csv" #"SimCatchCPUE_4.csv" #"Stocks_Catch_Test.csv"  #"Stock_Catch_forRainer.csv" # "SimCatchCPUE_3.csv" #  name of file containing "Stock", "yr", "ct", and optional "bt"
id_file     <- "cinfo_shrimp.csv" #"Stocks_ID_forDeng_RSb - Copy.csv" # "CombStocks_ID_2020CMSYrun3_v3_RF3.csv"   #    "SAUP_ID_2.csv"    #"SimSpecCPUE_4_NA_int_end.csv"  #"Train_ID_7j.csv" # "Stocks_ID_Aust_4.csv"  #"STECF_ID_2020_2.csv" #tRFMO_ID_2020_2.csv" #"ICES_ID_2020_4.csv" #"Robust_Stocks_ID_10_allNA.csv" #"SimSpecCPUE_4.csv"#"Stocks_ID_R_7.csv"  #"Stock_ID_forRainer.csv"  #  "NCod_ID_4.csv" #"SimSpecCPUE_3.csv" #  name of file containing stock-specific info and settings for the analysis
nn_file     <-  "ffnn.bin" # file containing neural networks trained to estimate B/k priors
outfile     <- paste("Out_",format(Sys.Date(),format="%B%d%Y_"),id_file,sep="") # default name for output file
out_list    <- list()

#some empty frames to store results
bio_out<-data.frame(stock=NULL,name=NULL,yr=NULL,region=NULL,scenario=NULL,source=NULL,bk_method=NULL,r_method=NULL,
    nyr=NULL,start.yr=NULL,int.yr=NULL,end.yr=NULL,
    ucl.B.Bmsy=NULL,lcl.B.Bmsy=NULL,B.Bmsy=NULL,
    ucl.F.Fmsy=NULL,lcl.F.Fmsy=NULL,F.Fmsy=NULL)

rk_out<- data.frame(
  stock=NULL,name=NULL,region=NULL,scenario=NULL,source=NULL,bk_method=NULL,r_method=NULL,
  priorr=NULL,priork=NULL,postr=NULL,postk=NULL)

cmsy_out <- data.frame(
  stock = NULL, name = NULL, region = NULL, scenario = NULL,
  source = NULL, bk_method = NULL, r_method = NULL,
  priorrlw= NULL,priorr = NULL, priorrup=NULL,
  postrlw=NULL, postr = NULL, postrup= NULL,
  priorklw=NULL, priork = NULL, priorkup= NULL, 
  postklw=NULL ,postk = NULL,postkup= NULL,
  priorbklw= NULL,priorbk = NULL,priorbkup= NULL, 
  postbklw=NULL,postbk = NULL, postbkup= NULL,
  priormsylw= NULL, priormsy= NULL, priormsyup= NULL,
  postmsylw= NULL, postmsy= NULL, postmsyup= NULL,
  fmsylw= NULL, fmsy= NULL, fmsyup= NULL,
  bmsylw= NULL, bmsy= NULL, bmsyup=NULL, 
  ctlast=NULL,ctavg5=NULL,
  ffmsylw=NULL,ffmsy=NULL,ffmsyup=NULL,
  bbmsylw=NULL,bbmsy=NULL, bbmsyup=NULL,
  PPVRr = NULL, RUr =NULL, OVLr = NULL, BCr =NULL,
  PPVRbk =NULL, RUbk =NULL, OVLbk =NULL, BCbk =NULL,
  viab_rate = NULL,
  mean.log.r = NULL, sd.log.r =NULL,
  mean.log.k = NULL, sd.log.k = NULL
)

kobe_out=data.frame(stock=NULL,name=NULL,region=NULL,scenario=NULL,
                   source =NULL, bk_method =NULL, r_method =NULL,
                   x.F_Fmsy=NULL, y.b_bmsy=NULL)


#pre-reading of catch series and info data about each stock
cdat         <- read.csv(catch_file, header=T, dec=".", stringsAsFactors = FALSE)
cinfo        <- read.csv(id_file, header=T, dec=".", stringsAsFactors = FALSE)
head(cdat)
head(cinfo)
#----------------------------------------
# Select stock to be analyzed ----
#----------------------------------------
# ***take all stocks****
stks<- unique(cinfo$Stock)
message("Stocks to be assessed:", "\n"); print(stks)

for (stk in stks) { #loop through stock picking
  
  stocks      <- stk
  
  #----------------------------------------------------
  #Updating the cinfo only if the stock is projection
  #----------------------------------------------------
  if (grepl("Projection", stk)) {
    
    message("\nUpdating cinfo for projection stock: ", stk)
    
    #Extract base name (everything before "_Projection")
    stk_base <- sub("_Projection.*", "", stk)
    
    # Find all related scenarios in cmsy_out
    cmsy_group <- cmsy_out %>%
      filter(grepl(paste0("^", stk_base, "_Base"), stock))
    
    if (nrow(cmsy_group) == 0) {
      warning("No scenario found for base stock: ", stk_base)
      next
    }
    
    # Scaling the metrics
    cmsy_scaled <- cmsy_group %>%
      mutate(
        RU_sc  = scales::rescale(RUr, to = c(0, 1)),
        OVL_sc = scales::rescale(OVLr, to = c(0, 1)),
        VR_sc  = scales::rescale(viab_rate, to = c(0, 1))
      )
    
    # Calculate final score
    cmsy_scored <- cmsy_scaled %>%
      mutate(final_score = 0.3 * RU_sc + 0.3 * OVL_sc + 0.4 * VR_sc)
    
    # Select the best scenario from the group
    best_scenario <- cmsy_scored %>%
      slice_max(final_score, n = 1, with_ties = FALSE)
    
    message("Best case scenario for ", stk_base, ": ", best_scenario$stock)
    
    # Update the cinfo to the corresponding projection scenario.
    cinfo <- cinfo %>%
      mutate(
        r.low = ifelse(Stock == stk, best_scenario$postrlw, r.low),
        r.hi  = ifelse(Stock == stk, best_scenario$postrup, r.hi),
        int.yr= ifelse(Stock == stk, 2015, int.yr), #fixing 2015 as the intermediary year
        intb.low= ifelse(Stock == stk, best_scenario$postbklw, intb.low),
        intb.hi= ifelse(Stock == stk, best_scenario$postbkup, intb.hi),
        endb.low = NA, # 2025 The biomass trend needs to be estimated by CMSY
        endb.hi  = NA # 2025 The biomass trend needs to be estimated by CMSY
      )
  }  
  
  
  #-----------------------------------------
  # General settings for the analysis ----
  #-----------------------------------------
  CV.C         <- 0.15  #><>MSY: Add Catch CV
  CV.cpue      <- 0.2 #><>MSY: Add minimum realistic cpue CV
  sigmaR       <- 0.1 # overall process error for CMSY; SD=0.1 is the default
  cor.log.rk   <- -0.76 # empirical value of log r-k correlation in 250 stocks analyzed with BSM (without r-k correlation), used only in graph
  rk.cor.beta  <- c(2.52,3.37) # beta.prior for rk cor+1
  nbk          <- 3 # Number of B/k priors to be used by BSM, with options 1 (first year), 2 (first & intermediate), 3 (first, intermediate & final bk priors)
  bt4pr        <- F # if TRUE, available abundance data are used for B/k prior settings
  auto.start   <- F # if TRUE, start year will be set to first year with intermediate catch to avoid ambiguity between low and high bimass if catches are very low
  ct_MSY.lim   <- 1.21  # ct/MSY.pr ratio above which B/k prior is assumed constant
  q.biomass.pr <- c(0.9,1.1) # if btype=="biomass" this is the prior range for q
  n            <- 5000 # number of points in multivariate cloud in graph panel (b)
  ni           <- 3 # iterations for r-k-startbiomass combinations, to test different variability patterns; no improvement seen above 3
  nab          <- 3 # recommended=5; minimum number of years with abundance data to run BSM
  bw           <- 3 # default bandwidth to be used by ksmooth() for catch data
  mgraphs      <- T # set to TRUE to produce additional graphs for management
  e.creep.line <- T # set to TRUE to display uncorrected CPUE in biomass graph
  kobe.plot    <- T # set to TRUE to produce additional kobe status plot; management graph needs to be TRUE for Kobe to work
  BSMfits.plot <- F # set to TRUE to plot fit diagnostics for BSM
  pp.plot      <- T # set to TRUE to plot Posterior and Prior distributions for CMSY and BSM
  rk.diags     <- T #><>MSY set to TRUE to plot diagnostic plot for r-k space
  retros       <- F # set to TRUE to enable retrospective analysis (1-3 years less in the time series)
  save.plots   <- T # set to TRUE to save graphs to JPEG files
  close.plots  <- if (length(stks)>5) {close.plots=T}else {F} # set to TRUE to close on-screen plots after they are saved, to avoid "too many open devices" error in batch-processing
  write.output <- T # set to TRUE if table with results in output file is wanted; expects years 2004-2014 to be available
  write.pdf    <- F # set to TRUE if PDF output of results is wanted. See more instructions at end of code.
  select.yr    <- NA # option to display F, B, F/Fmsy and B/Bmsy for a certain year; default NA
  write.rdata  <- F #><>HW write R data file
  
  #----------------------------------------------
  #  FUNCTIONS ----
  #----------------------------------------------
  #------------------------------------------------------------------------------------
  # Function to create multivariate-normal distribution for r-k, used only in graphs
  #------------------------------------------------------------------------------------
  mvn   <- function(n,mean.log.r,sd.log.r,mean.log.k,sd.log.k) {
    cov.log.rk <- cor.log.rk*sd.log.r*sd.log.k # covariance with empirical correlation and prior variances  covar.log.rk = matrix(NA, ncol=2,nrow=2)   # contract covariance matrix
    covar.log.rk      <- matrix(NA, ncol=2,nrow=2) # covariance matrix
    covar.log.rk[1,1] <- sd.log.r^2                # position [1,1] is variance of log.r
    covar.log.rk[2,2] <- sd.log.k^2               # position [2,2] is variance of log.k
    covar.log.rk[1,2] = covar.log.rk[2,1] = cov.log.rk     # positions [1,2] and [2,1] are correlations
    mu.log.rk  <- (c(mean.log.r,mean.log.k))      # vector of log.means
    mvn.log.rk <- rmvnorm(n,mean=mu.log.rk,sigma=covar.log.rk,method="svd")
    return(mvn.log.rk)
  }
  
  #-------------------------------------------------------------
  # Function to run Bayesian Schaefer Model (BSM)
  #-------------------------------------------------------------
  bsm   <- function(ct,btj,nyr,prior.r,prior.k,startbio,q.priorj,
    init.q,init.r,init.k,pen.bk,pen.F,b.yrs,b.prior,CV.C,CV.cpue,nbk,rk.cor.beta,cmsyjags) {
    #><> convert b.prior ranges into beta priors
    bk.beta = beta.prior(b.prior)
    
    if(cmsyjags==TRUE ){ nbks=3 } else {nbks = nbk} # Switch between CMSY + BSM
    
    # Data to be passed on to JAGS
    jags.data        <- c('ct','btj','nyr', 'prior.r', 'prior.k', 'startbio', 'q.priorj',
      'init.q','init.r','init.k','pen.bk','pen.F','b.yrs','bk.beta','CV.C','CV.cpue','nbks','rk.cor')
    # Parameters to be returned by JAGS #><> HW add key quantities
    jags.save.params <- c('r','k','q', 'P','ct.jags','cpuem','proc.logB','B','F','BBmsy','FFmsy','ppd.logrk')
    
    # JAGS model ----
    Model = "model{
    # to reduce chance of non-convergence, Pmean[t] values are forced >= eps
    eps<-0.01
    #><> Add Catch.CV
    for(t in 1:nyr){
      ct.jags[t] ~ dlnorm(log(ct[t]),pow(CV.C,-2))
    }

    penm[1]  <- 0 # no penalty for first biomass
    Pmean[1] <- log(alpha)
    P[1]     ~ dlnorm(Pmean[1],itau2)

    for (t in 2:nyr) {
      Pmean[t] <- ifelse(P[t-1] > 0.25,
        log(max(P[t-1] + r*P[t-1]*(1-P[t-1]) - ct.jags[t-1]/k,eps)),  # Process equation
        log(max(P[t-1] + 4*P[t-1]*r*P[t-1]*(1-P[t-1]) - ct.jags[t-1]/k,eps))) # linear decline of r at B/k < 0.25
      P[t]     ~ dlnorm(Pmean[t],itau2) # Introduce process error
      penm[t]  <- ifelse(P[t]<(eps+0.001),log(q*k*P[t])-log(q*k*(eps+0.001)),
                   # ifelse(P[t]>1,ifelse((ct[t]/max(ct))>0.2,log(q*k*P[t])-log(q*k*(0.99)),0),0)) # penalty if Pmean is outside viable biomass
                    ifelse(P[t]>1.1,log(q*k*P[t])-log(q*k*(0.99)),0))
    }

    # Get Process error deviation
    for(t in 1:nyr){
      proc.logB[t] <- log(P[t]*k)-log(exp(Pmean[t])*k)}

    # ><> b.priors with penalties
    # Biomass priors/penalties are enforced as follows
    for(i in 1:nbks){
    bk.mu[i] ~ dbeta(bk.beta[1,i],bk.beta[2,i])
    bk.beta[3,i] ~ dnorm(bk.mu[i]-P[b.yrs[i]],10000)
    }

    for (t in 1:nyr){
      Fpen[t]   <- ifelse(ct[t]>(0.9*k*P[t]),ct[t]-(0.9*k*P[t]),0) # Penalty term on F > 1, i.e. ct>B
      pen.F[t]  ~ dnorm(Fpen[t],1000)
      pen.bk[t] ~ dnorm(penm[t],10000)
      cpuem[t]  <- log(q*P[t]*k);
      btj[t]     ~ dlnorm(cpuem[t],pow(sigma2,-1));
    }

  # priors
  log.alpha               <- log((startbio[1]+startbio[2])/2) # needed for fit of first biomass
  sd.log.alpha            <- (log.alpha-log(startbio[1]))/4
  tau.log.alpha           <- pow(sd.log.alpha,-2)
  alpha                   ~  dlnorm(log.alpha,tau.log.alpha)

  # set realistic prior for q
  log.qm              <- mean(log(q.priorj))
  sd.log.q            <- (log.qm-log(q.priorj[1]))/2
  tau.log.q           <- pow(sd.log.q,-2)
  q                   ~  dlnorm(log.qm,tau.log.q)

  # define process (tau) and observation (sigma) variances as inversegamma priors
  itau2 ~ dgamma(4,0.01)
  tau2  <- 1/itau2
  tau   <- pow(tau2,0.5)

  isigma2 ~ dgamma(2,0.01)
  sigma2 <- 1/isigma2+pow(CV.cpue,2) # Add minimum realistic CPUE CV
  sigma  <- pow(sigma2,0.5)

  log.rm              <- mean(log(prior.r))
  sd.log.r            <- abs(log.rm - log(prior.r[1]))/2
  tau.log.r           <- pow(sd.log.r,-2)

  # bias-correct lognormal for k
  log.km              <- mean(log(prior.k))
  sd.log.k            <- abs(log.km-log(prior.k[1]))/2
  tau.log.k           <- pow(sd.log.k,-2)

  # Construct Multivariate lognormal (MVLN) prior
  mu.rk[1] <- log.rm
  mu.rk[2] <- log.km

  # Prior for correlation log(r) vs log(k)
  #><>MSY: now directly taken from mvn of ki = 4*msyi/ri
  rho <- rk.cor

  # Construct Covariance matrix
  cov.rk[1,1] <- sd.log.r * sd.log.r
  cov.rk[1,2] <- rho
  cov.rk[2,1] <- rho
  cov.rk[2,2] <- sd.log.k * sd.log.k

  # MVLN prior for r-k
  log.rk[1:2] ~ dmnorm(mu.rk[],inverse(cov.rk[,]))
  r <- exp(log.rk[1])
  k <- exp(log.rk[2])

  #><>MSY get posterior predictive distribution for rk
  ppd.logrk[1:2] ~ dmnorm(mu.rk[],inverse(cov.rk[,]))

  # ><>HW: Get B/Bmsy and F/Fmsy directly from JAGS
  Bmsy <- k/2
  Fmsy <- r/2
  for (t in 1:nyr){
  B[t] <- P[t]*k # biomass
  F[t] <- ct.jags[t]/B[t]
  BBmsy[t] <- P[t]*2 #true for Schaefer
  FFmsy[t] <- ifelse(BBmsy[t]<0.5,F[t]/(Fmsy*2*BBmsy[t]),F[t]/Fmsy)
  }
} "    # end of JAGS model
    
    # Write JAGS model to file ----
    cat(Model, file="r2jags.bug")
    
    #><>MSY: change to lognormal inits (better)
    j.inits <- function(){list("log.rk"=c(rnorm(1,mean=log(init.r),sd=0.2),rnorm(1,mean=log(init.k),sd=0.1)),
      "q"=rlnorm(1,mean=log(init.q),sd=0.2),"itau2"=1000,"isigma2"=1000)}
    # run model ----
    jags_outputs <- jags.parallel(data=jags.data,
      working.directory=NULL, inits=j.inits,
      parameters.to.save=jags.save.params,
      model.file="r2jags.bug", n.chains = n.chains,
      n.burnin = 30000, n.thin = 10,
      n.iter = 60000)
    return(jags_outputs)
  }
  
  #><> beta.prior function
  get_beta <- function(mu,CV,Min=0,Prior="x",Plot=FALSE){
    a = seq(0.0001,1000,0.001)
    b= (a-mu*a)/mu
    s2 = a*b/((a+b)^2*(a+b+1))
    sdev = sqrt(s2)
    # find beta parameter a
    CV.check = (sdev/mu-CV)^2
    a = a[CV.check==min(CV.check)]
    # find beta parameter b
    b = (a-mu*a)/mu
    x = seq(Min,1,0.001)
    pdf = dbeta(x,a,b)
    if(Plot==TRUE){
      plot(x,pdf,type="l",xlim=range(x[pdf>0.01]),xlab=paste(Prior),ylab="",yaxt="n")
      polygon(c(x,rev(x)),c(rep(0,length(x)),rev(ifelse(pdf==Inf,100000,pdf))),col="grey")
    }
    return(c(a,b))
  }
  
  #><> convert b.prior ranges into beta priors
  beta.prior = function(b.prior){
    bk.beta = matrix(0,nrow = 3,ncol=3)
    for(i in 1:3){
      sd.bk = (b.prior[2,i]-b.prior[1,i])/(4*0.98)
      mu.bk = mean(b.prior[1:2,i])
      cv.bk = sd.bk/mu.bk
      bk.beta[1:2,i] = get_beta(mu.bk,cv.bk)
    }
    return(bk.beta)
  }
  
  #Fits an ellipse around the CMSY r-k cloud and estimates the rightmost focus
  traceEllipse<-function(rs,ks,prior.r,prior.k){
    log.rs<-log(rs)
    log.ks<-log(ks)
    
    #  #select data within the bounding box
    #  log.rs<-log.rs[which(rs>prior.r[1] & rs<prior.r[2] &
    #                         ks>prior.k[1] & ks<prior.k[2]
    #  )]
    #  log.ks<-log.ks[which(rs>prior.r[1] & rs<prior.r[2] &
    #                         ks>prior.k[1] & ks<prior.k[2]
    #  )]
    
    #prepare data for ellipse fitting
    cloud.data <- as.matrix(data.frame(x = log.rs, y = log.ks))
    ellip <- EllipseDirectFit(cloud.data)
    #estimate ellipse characteristics
    atog<-AtoG(ellip)
    ellipG <- atog$ParG
    ell.center.x<-ellipG[1]
    ell.center.y<-ellipG[2]
    ell.axis.a<-ellipG[3]
    ell.axis.b<-ellipG[4]
    ell.tilt.angle.deg<-180/pi*ellipG[5]
    ell.slope<-tan(ellipG[5])
    xy.ell<-calculateEllipse(ell.center.x,
      ell.center.y,
      ell.axis.a,
      ell.axis.b,
      ell.tilt.angle.deg)
    #draw ellipse
    #points(x=xy.ell[,1],y=xy.ell[,2],col='red',type='l')
    ell.intercept.1 = ell.center.y-ell.center.x*ell.slope
    #draw ellipse main axis
    #abline(a =ell.intercept.1, b=ell.slope,col='red')
    #calculate focus from demi-axes
    ell.demiaxis.c.sqr<-(0.25*ell.axis.a*ell.axis.a)-(0.25*ell.axis.b*ell.axis.b)
    if (ell.demiaxis.c.sqr<0)
      ell.demiaxis.c.sqr<-ell.axis.a/2
    else
      ell.demiaxis.c<-sqrt(ell.demiaxis.c.sqr)
    sin.c<-ell.demiaxis.c*sin(ellipG[5])
    cos.c<-ell.demiaxis.c*cos(ellipG[5])
    ell.foc.y<-ell.center.y-sin.c
    ell.foc.x<-ell.center.x-cos.c
    #draw focus
    #points(x=ell.foc.x,y=ell.foc.y,
    #      pch = 16, cex = 1.2,
    #     col='green',bty='l')
    
    return (c(exp(ell.foc.x),exp(ell.foc.y)))
  }
  #---------------------------------------------
  # END OF FUNCTIONS
  #---------------------------------------------
  
  #-----------------------------------------
  # Start output to screen
  #-----------------------------------------
  cat("-------------------------------------------\n")
  cat("CMSY++ Analysis,", date(),"\n")
  cat("-------------------------------------------\n")
  
  #------------------------------------------
  # Read data and assign to vectors
  #------------------------------------------
  # create headers for data table file
  if(write.output==T){
    outheaders = data.frame("Group","Region", "Subregion","Name","SciName","Stock",
      "start.yr","end.yr","start.yr.new","btype",
      "N bt","start.yr.cpue","end.yr.cpue","min.cpue","max.cpue","min.yr.cpue","max.yr.cpue",
      "endbio.low","endbio.hi","q.prior.low","q.prior.hi",
      "MaxCatch","MSY_prior","MeanLast5RawCatch","SDLast5RawCatch","LastCatch",
      "MinSmoothCatch","MaxSmoothCatch","MeanSmoothCatch","gMeanPrior_r",
      "MSY_BSM","lcl.MSY_BSM","ucl.MSY_BSM","r_BSM","lcl.r_BSM","ucl.r_BSM","log.r_var",
      "k_BSM","lcl.k_BSM","ucl.k_BSM","log.k_var","log.kr_cor","log.kr_cov","q_BSM","lcl.q_BSM","ucl.q_BSM",
      "rel_B_BSM","lcl.rel_B_BSM","ucl.rel_B_BSM","rel_start_B_BSM","lcl.rel_start_B_BSM","ucl.rel_start_B_BSM",
      "rel_int_B_BSM","lcl.rel_int_B_BSM","ucl.rel_int_B_BSM","int.yr","rel_F_BSM",
      "r_CMSY","lcl.r_CMSY","ucl.r_CMSY","k_CMSY","lcl.k_CMSY","ucl.k_CMSY","MSY_CMSY","lcl.MSY_CMSY","ucl.MSY_CMSY",
      "rel_B_CMSY","2.5th.rel_B_CMSY","97.5th.rel_B_CMSY","rel_start_B_CMSY","2.5th.rel_start_B_CMSY","97.5th.rel_start_B_CMSY",
      "rel_int_B_CMSY","2.5th.rel_int_B_CMSY","97.5th.rel_int_B_CMSY",
      "rel_F_CMSY","2.5th.rel_F_CMSY","97.5th.rel_F_CMSY",
      "F_msy","lcl.F_msy","ucl.F_msy","curF_msy","lcl.curF_msy","ucl.curF_msy",
      "MSY","lcl.MSY","ucl.MSY","Bmsy","lcl.Bmsy","ucl.Bmsy",
      "last.B","lcl.last.B","ucl.last.B","last.B_Bmsy","lcl.last.B_Bmsy","ucl.last.B_Bmsy",
      "last.F","lcl.last.F","ucl.last.F","last.F_Fmsy","lcl.last.F_Fmsy","ucl.last.F_Fmsy",
      "sel_B","sel_B_Bmsy","sel_F","sel_F_Fmsy",
      # create columns for catch, F/Fmsy and Biomass for 1950 to 2020
      "c50","c51","c52","c53","c54","c55","c56","c57","c58","c59",
      "c60","c61","c62","c63","c64","c65","c66","c67","c68","c69",
      "c70","c71","c72","c73","c74","c75","c76","c77","c78","c79",
      "c80","c81","c82","c83","c84","c85","c86","c87","c88","c89",
      "c90","c91","c92","c93","c94","c95","c96","c97","c98","c99",
      "c00","c01","c02","c03","c04","c05","c06","c07","c08","c09",
      "c10","c11","c12","c13","c14","c15","c16","c17","c18","c19",
      "c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30",
      "F.Fmsy50","F.Fmsy51","F.Fmsy52","F.Fmsy53","F.Fmsy54","F.Fmsy55","F.Fmsy56","F.Fmsy57","F.Fmsy58","F.Fmsy59",
      "F.Fmsy60","F.Fmsy61","F.Fmsy62","F.Fmsy63","F.Fmsy64","F.Fmsy65","F.Fmsy66","F.Fmsy67","F.Fmsy68","F.Fmsy69",
      "F.Fmsy70","F.Fmsy71","F.Fmsy72","F.Fmsy73","F.Fmsy74","F.Fmsy75","F.Fmsy76","F.Fmsy77","F.Fmsy78","F.Fmsy79",
      "F.Fmsy80","F.Fmsy81","F.Fmsy82","F.Fmsy83","F.Fmsy84","F.Fmsy85","F.Fmsy86","F.Fmsy87","F.Fmsy88","F.Fmsy89",
      "F.Fmsy90","F.Fmsy91","F.Fmsy92","F.Fmsy93","F.Fmsy94","F.Fmsy95","F.Fmsy96","F.Fmsy97","F.Fmsy98","F.Fmsy99",
      "F.Fmsy00","F.Fmsy01","F.Fmsy02","F.Fmsy03","F.Fmsy04","F.Fmsy05","F.Fmsy06","F.Fmsy07","F.Fmsy08","F.Fmsy09",
      "F.Fmsy10","F.Fmsy11","F.Fmsy12","F.Fmsy13","F.Fmsy14","F.Fmsy15","F.Fmsy16","F.Fmsy17","F.Fmsy18","F.Fmsy19",
      "F.Fmsy20","F.Fmsy21","F.Fmsy22","F.Fmsy23","F.Fmsy24","F.Fmsy25","F.Fmsy26","F.Fmsy27","F.Fmsy28","F.Fmsy29","F.Fmsy30",
      "B50","B51","B52","B53","B54","B55","B56","B57","B58","B59",
      "B60","B61","B62","B63","B64","B65","B66","B67","B68","B69",
      "B70","B71","B72","B73","B74","B75","B76","B77","B78","B79",
      "B80","B81","B82","B83","B84","B85","B86","B87","B88","B89",
      "B90","B91","B92","B93","B94","B95","B96","B97","B98","B99",
      "B00","B01","B02","B03","B04","B05","B06","B07","B08","B09",
      "B10","B11","B12","B13","B14","B15","B16","B17","B18","B19",
      "B20","B21","B22","B23","B24","B25","B26","B27","B28","B29","B30")
    write.table(outheaders,file=outfile, append = T, sep=",",row.names=F,col.names=F)
  }
  
  # Read data
  cdat         <- cdat#read.csv(catch_file, header=T, dec=".", stringsAsFactors = FALSE)
  cinfo        <- cinfo#read.csv(id_file, header=T, dec=".", stringsAsFactors = FALSE)
  load(file = nn_file) # load neural network file
  cat("Files", catch_file, ",", id_file,",",nn_file,"read successfully","\n")
  
  #---------------------------------
  # Analyze stock(s)
  #---------------------------------
  if(is.na(stocks[1])==TRUE){
    # stocks         <- as.character(cinfo$Stock) # Analyze stocks in sequence of ID file
    # stocks         <- sort(as.character(cinfo$Stock[cinfo$Stock>="Cras_vir_Virginian"])) # Analyze in alphabetic order after a certain stock
    stocks         <- sort(as.character(cinfo$Stock)) # Analyze stocks in alphabetic order
    # stocks         <- as.character(cinfo$Stock[cinfo$btype!="None" & cinfo$Stock>"Squa_aca_BlackSea"]) # Analyze stocks by criteria in ID file
  }
  
  # analyze one stock after the other...
  for(stock in stocks) {
    
    cat("Processing",stock,",", as.character(cinfo$ScientificName[cinfo$Stock==stock]),"\n")
    
    #retrospective analysis
    retros.nyears<-ifelse(retros==T,3,0) #retrospective analysis
    FFmsy.retrospective<-list() #retrospective analysis
    BBmsy.retrospective<-list() #retrospective analysis
    years.retrospective<-list() #retrospective analysis
    
    retrosp.step =0
    for (retrosp.step in 0:retros.nyears){ #retrospective analysis loop
      
      # Declare conditional Objects that feature with ifelse clauses
      B.sel        <- NULL
      B.Bmsy.sel   <- NULL
      F.sel        <- NULL
      F.Fmsy.sel   <- NULL
      true.MSY     <- NULL
      true.r       <- NULL
      true.k       <- NULL
      true.Bk      <- NULL
      true.F_Fmsy  <- NULL
      true.q       <- NULL
      
      # assign data from cinfo to vectors
      btype        <- as.character(cinfo$btype[cinfo$Stock==stock])
      res          <- as.character(cinfo$Resilience[cinfo$Stock==stock])
      start.yr     <- as.numeric(cinfo$StartYear[cinfo$Stock==stock])
      end.yr       <- as.numeric(cinfo$EndYear[cinfo$Stock==stock])
      end.yr.orig  <- end.yr
      end.yr 	     <- end.yr-retrosp.step #retrospective analysis
      yr           <- as.numeric(cdat$yr[cdat$Stock==stock & cdat$yr >= start.yr & cdat$yr <= end.yr])
      if(length(yr)==0){
        cat("ERROR: Could not find the stock in the Catch file -
      check that the stock names match in ID and Catch files and that commas are used (not semi-colon)")
        return (NA) }
      
      # code to change start year to avoid ambiguity in biomass prior -----------------------------------------
      ct.raw       <- as.numeric(cdat$ct[cdat$Stock==stock & cdat$yr >= start.yr & cdat$yr <= end.yr])/1000  ## assumes that catch is given in tonnes, transforms to '000 tonnes
      ct           <- ksmooth(x=yr,y=ct.raw,kernel="normal",n.points=length(yr),bandwidth=bw)$y
      ct.3         <- mean(ct[1:3])
      max.ct       <- max(ct)
      
      if(btype=="biomass" | btype=="CPUE" ) {
        bt.raw1 <- as.numeric(cdat$bt[cdat$Stock==stock & cdat$yr >= start.yr & cdat$yr <= end.yr])
        # if bt.raw is zero, change to NA
        bt.raw1[bt.raw1==0] <- NA
        if(btype=="biomass") { # make sure both catch and biomass are divided by 1000
          bt <- bt.raw1/1000 } else { # get number of integer digits for bt.raw (because sometimes they give numbers of eggs!)
            bt.digits <- floor(log10(mean(bt.raw1,na.rm=T)))+1
            if(bt.digits>3) {bt.raw <- bt.raw1/10^(bt.digits-1)} else {bt.raw <- bt.raw1}
            bt     <- bt.raw #ksmooth(x=yr,y=bt.raw,kernel="normal",n.points=length(yr),bandwidth=3)$y
          } # end of bt==CPUE loop
        if(length(bt[is.na(bt)==F])==0) {
          cat("ERROR: No CPUE or biomass data in the Catch input file")
          return (NA) }
      } else {bt <- NA; bt.raw <- NA} # if there is no biomass or CPUE, set bt to NA
      
      
      # code to change start year to avoid ambiguity in biomass prior -----------------------------------------
      start.yr.new <- NA # initialize / reset start.yr.new with NA
      if(is.na(cinfo$stb.low[cinfo$Stock==stock]) & ct.3 < (0.33*max.ct) & start.yr < 2000 & (btype=="None" || yr[is.na(bt)==F][1]>yr[3])) { # it is unlikely that a fishery started on an unexploited stock after 2000
        start.yr.new <- yr[which(ct >= (0.4*max.ct))][1]
        cat("\n          *****************************************************************************************
          Attention: Low catch in",start.yr,"may indicate either depleted or unexploited biomass.
          Set startbio in ID file to 0.01-0.2 or 0.8-1.0 to indicate depleted or unexploited biomass.\n")
        if(auto.start==T) { # change start year automatically if auto.start is TRUE
          start.yr  <- start.yr.new
          cat("          Meanwhile start year was set to",start.yr,"to avoid ambiguity.\n")
        } else {
          cat("          Else, set start year in ID file to",start.yr.new,"to avoid uncertainty\n")
        }
        cat("          ******************************************************************************************\n\n") }
      # end of code for start biomass prior ambiguity
      
      ename        <- cinfo$Name[cinfo$Stock==stock]
      r.low        <- as.numeric(cinfo$r.low[cinfo$Stock==stock])
      r.hi         <- as.numeric(cinfo$r.hi[cinfo$Stock==stock])
      stb.low      <- as.numeric(cinfo$stb.low[cinfo$Stock==stock])
      stb.hi       <- as.numeric(cinfo$stb.hi[cinfo$Stock==stock])
      int.yr       <- as.numeric(cinfo$int.yr[cinfo$Stock==stock])
      intb.low     <- as.numeric(cinfo$intb.low[cinfo$Stock==stock])
      intb.hi      <- as.numeric(cinfo$intb.hi[cinfo$Stock==stock])
      endb.low     <- as.numeric(cinfo$endb.low[cinfo$Stock==stock])
      endb.hi      <- as.numeric(cinfo$endb.hi[cinfo$Stock==stock])
      e.creep      <- as.numeric(cinfo$e.creep[cinfo$Stock==stock])
      force.cmsy   <- cinfo$force.cmsy[cinfo$Stock==stock]
      comment      <- as.character(cinfo$Comment[cinfo$Stock==stock])
      source       <- as.character(cinfo$Source[cinfo$Stock==stock])
      # set global defaults for uncertainty
      sigR         <- sigmaR
      # for simulated data only
      if(substr(id_file,1,3)=="Sim") {
        true.MSY     <- cinfo$true.MSY[cinfo$Stock==stock]/1000
        true.r       <- cinfo$true.r[cinfo$Stock==stock]
        true.k       <- cinfo$true.k[cinfo$Stock==stock]/1000
        true.Bk      <- (cinfo$last.TB[cinfo$Stock==stock]/1000)/true.k
        true.F_Fmsy  <- cinfo$last.F_Fmsy[cinfo$Stock==stock]
        true.q       <- cinfo$last.cpue[cinfo$Stock==stock]/cinfo$last.TB[cinfo$Stock==stock]
      }
      # do retrospective analysis
      if (retros==T && retrosp.step==0){
        cat("* ",ifelse(btype!="None","BSM","CMSY")," retrospective analysis for ",
          stock," has been enabled\n",sep="") #retrospective analysis
      }
      if (retros==T){
        cat("* Retrospective analysis: step n. ",(retrosp.step+1),"/",(retros.nyears+1),
          ". Range of years: [",start.yr ," - ",end.yr,"]\n",sep="") #retrospective analysis
      }
      
      # -------------------------------------------------------------
      # check for common errors
      #--------------------------------------------------------------
      if(length(btype)==0){
        cat("ERROR: Could not find the stock in the ID input file - check that the stock names match in ID and Catch files and that commas are used (not semi-colon)")
        return (NA) }
      if(start.yr < cdat$yr[cdat$Stock==stock][1]){
        cat("ERROR: start year in ID file before first year in catch file\n")
        return (NA)
        break}
      if(length(yr)==0){
        cat("ERROR: Could not find the stock in the Catch input files - Please check that the code is written correctly")
        return (NA) }
      if(btype %in% c("None","CPUE","biomass")==FALSE){
        cat("ERROR: In ID file, btype must be None, CPUE, or biomass.")
        return (NA) }
      if(retros==F & length(yr) != (end.yr-start.yr+1)) {
        cat("ERROR: indicated year range is of different length than years in catch file\n")
        return (NA)}
      if(length(ct.raw[ct.raw>0])==0) {
        cat("ERROR: No catch data in the Catch input file")
        #return (NA)
        next }
      if(is.na(int.yr)==F & (int.yr < start.yr | int.yr > end.yr)) {
        cat("ERROR: year for intermediate B/k prior outside range of years")
        return (NA)}
      if(is.na(int.yr)==T & (is.na(intb.low)==F | is.na(intb.hi)==F)) {
        cat("ERROR: intermediate B/k prior given without year")
        return (NA)}
      
      # apply correction for effort-creep to commercial(!) CPUE
      if(btype=="CPUE" && is.na(e.creep)==FALSE) {
        cpue.first  <- min(which(is.na(bt)==F))
        cpue.last   <- max(which(is.na(bt)==F))
        cpue.length <- cpue.last - cpue.first
        bt.cor      <- bt
        for(i in 1:(cpue.length)) {
          bt.cor[cpue.first+i]  <- bt[cpue.first+i]*(1-e.creep/100)^i # equation for decay in %
        }
        bt <- bt.cor
      }
      
      if(retros==T && force.cmsy == F && (btype !="None" & length(bt[is.na(bt)==F])<nab) ) { #stop retrospective analysis if cpue is < nab
        cat("Warning: Cannot run retrospective analysis for ",end.yr,", number of remaining ",btype," values is too low (<",nab,")\n",sep="")
        #retrosp.step<-retros.nyears
        break }
      
      if(is.na(mean(ct.raw))){
        cat("ERROR: Missing value in Catch data; fill or interpolate\n")
      }
      nyr          <- length(yr) # number of years in the time series
      
      
      # initialize vectors for viable r, k, bt, and all in a matrix
      mdat.all    <- matrix(data=vector(),ncol=2+nyr+1)
      
      # initialize other vectors anew for each stock
      current.attempts <- NA
      
      # use start.yr if larger than select year
      if(is.na(select.yr)==F) {
        sel.yr <- ifelse(start.yr > select.yr,start.yr,select.yr)
      } else sel.yr <- NA
      
      #----------------------------------------------------
      # Determine initial ranges for parameters and biomass
      #----------------------------------------------------
      if(!(res %in% c("High","Medium","Low","Very low"))) {
        cat("ERROR: Resilience not High, Medium, Low, or Very low in ID input file")
        return (NA)} else {
          # initial range of r from input file
          if(is.na(r.low)==F & is.na(r.hi)==F) {
            prior.r <- c(r.low,r.hi)
          } else
            # initial range of r based on resilience
            if(res == "High") {
              prior.r <- c(0.6,1.5)} else if(res == "Medium") {
                prior.r <- c(0.2,0.8)}    else if(res == "Low") {
                  prior.r <- c(0.05,0.5)}  else { # i.e. res== "Very low"
                    prior.r <- c(0.015,0.1)}
        }
      gm.prior.r      <- exp(mean(log(prior.r))) # get geometric mean of prior r range
      
      #-----------------------------------------
      # determine MSY prior
      #-----------------------------------------
      # get index of years with lowest and highest catch
      min.yr.i     <- which.min(ct)
      max.yr.i     <- which.max(ct)
      yr.min.ct    <- yr[min.yr.i]
      yr.max.ct    <- yr[max.yr.i]
      min.ct       <- ct[min.yr.i]
      max.ct       <- ct[max.yr.i]
      min_max      <- min.ct/max.ct
      mean.ct      <- mean(ct)
      sd.ct        <- sd(ct)
      
      ct.sort     <- sort(ct.raw)
      # if max catch is reached in last 5 years or catch is flat, assume MSY=max catch
      if(max.yr.i>(nyr-4) || ((sd.ct/mean.ct) < 0.1 && min_max > 0.66)) {
        MSY.pr <- mean(ct.sort[(nyr-2):nyr]) } else {
          MSY.pr <- 0.75*mean(ct.sort[(nyr-4):nyr]) } # else, use fraction of mean of 5 highest catches as MSY prior
      
      #><>MSY: MSY prior
      sd.log.msy.pr <- 0.3 # rounded upward to account for reduced variability in selected stocks
      log.msy.pr    <- log(MSY.pr)
      prior.msy     <- c(exp(log.msy.pr-1.96*sd.log.msy.pr),exp(log.msy.pr+1.96*sd.log.msy.pr))
      init.msy      <- MSY.pr
      
      #----------------------------------------------------------------
      # Multivariate normal sampling of r-k log space
      #----------------------------------------------------------------
      # turn numerical ranges into log-normal distributions
      mean.log.r=mean(log(prior.r))
      sd.log.r=(log(prior.r[2])-log(prior.r[1]))/(2*1.96)  # assume range covers 4 SD
      
      #><>MSY: new k = r-msy space
      # generate msy and r independently
      ri1     <- rlnorm(n,mean.log.r,sd.log.r)
      msyi1  <- rlnorm(n,log.msy.pr,sd.log.msy.pr)
      ki1     <- msyi1*4/ri1
      #><>MSY: get log median and covariance
      cov_rk <- cov(cbind(log(ri1),log(ki1)))
      mu_rk <-  apply(cbind(log(ri1),log(ki1)),2,median)
      rk.cor <- cov_rk[2,1] #MSY: correlation rho input to JAGS
      #><>MSY: mvn prior for k = 4*msy/r
      mvn.log.rk <- rmvnorm(n,mean=mu_rk,cov_rk)
      
      ri2    <- exp(mvn.log.rk[,1])
      ki2    <- exp(mvn.log.rk[,2])
      
      mean.log.k <- median(log(ki1))
      sd.log.k.pr <- sd(log(ki1))
      # quick check must be the same
      sd.log.k = sqrt(cov_rk[2,2])
      sd.log.k.pr
      sd.log.k
      #><>MSY: k.prior
      prior.k     <- exp(mean.log.k-1.96*sd.log.k.pr) # declare variable and set prior.k[1] in one step
      prior.k[2]  <- exp(mean.log.k+1.96*sd.log.k.pr)
      msy.init <- exp(mean.log.k)
      
      #-----------------------------------------
      # determine prior B/k ranges
      #-------------------------------------------------
      # determine intermediate year int.yr for prior B/k
      if(is.na(cinfo$int.yr[cinfo$Stock==stock])==F) {
        int.yr <- cinfo$int.yr[cinfo$Stock==stock]     # use int.yr give by user
      } else {if(min_max > 0.7) { # if catch is about flat, use middle year as int.yr
        int.yr    <- as.integer(mean(c(start.yr, end.yr)))
      } else { # only consider catch 5 years away from end points and within last 30 years # 50
        yrs.int       <- yr[yr>(yr[nyr]-30) & yr>yr[4] & yr<yr[nyr-4]]
        ct.int        <- ct[yr>(yr[nyr]-30) & yr>yr[4] & yr<yr[nyr-4]]
        min.ct.int    <- min(ct.int)
        min.ct.int.yr <- yrs.int[which.min(ct.int)]
        max.ct.int    <- max(ct.int)
        max.ct.int.yr <- yrs.int[which.max(ct.int)]
        #if min year is after max year, use min year for int year
        if(min.ct.int.yr > max.ct.int.yr) { int.yr <- min.ct.int.yr } else {
          # if min.ct/max.ct after max.ct < 0.7, use that year for int.yr
          min.ct.after.max <- min(ct.int[yrs.int >= max.ct.int.yr])
          if((min.ct.after.max/max.ct.int) < 0.75) {
            int.yr <- yrs.int[yrs.int > max.ct.int.yr & ct.int==min.ct.after.max]
          } else {int.yr <- min.ct.int.yr}
        }
        # get latest year where ct < 1.2 min ct
        # int.yr        <- max(yrs.int[ct.int<=(1.2*min.ct.int)])
      }
      }# end of int.yr loop
      
      # get additional properties of catch time series
      mean.ct.end       <- mean(ct.raw[(nyr-4):nyr]) # mean of catch in last 5 years
      mean.ct_MSY.end   <- mean.ct.end/MSY.pr
      # Get slope of catch in last 10 years
      ct.last           <- ct[(nyr-9):nyr]/mean(ct) # last catch standardized by mean catch
      yrs.last          <- seq(1:10)
      fit.last          <- lm(ct.last ~ yrs.last)
      slope.last        <- as.numeric(coefficients(fit.last)[2])
      slope.last.nrm    <- (slope.last - slope.last.min)/(slope.last.max - slope.last.min) # normalized slope 0-1
      # Get slope of catch in first 10 years
      ct.first          <- ct[1:10]/mean.ct # catch standardized by mean catch
      yrs.first         <- seq(1:10)
      fit.first         <- lm(ct.first ~ yrs.first)
      slope.first       <- as.numeric(coefficients(fit.first)[2])
      slope.first.nrm   <- (slope.first - slope.first.min)/(slope.first.max - slope.first.min) # normalized slope 0-1
      
      ct_max.1          <- ct[1]/max.ct
      ct_MSY.1          <- ct[1]/MSY.pr
      mean.ct_MSY.start <- mean(ct.raw[1:5])/MSY.pr
      ct_MSY.int        <- ct[which(yr==int.yr)]/MSY.pr
      ct_max.end        <- ct[nyr]/max.ct
      ct_MSY.end        <- ct[nyr]/MSY.pr
      max.ct.i          <- which.max(ct)/nyr
      int.ct.i          <- which(yr==int.yr)/nyr
      min.ct.i          <- which.min(ct)/nyr
      yr.norm           <- (nyr - yr.norm.min)/(yr.norm.max - yr.norm.min) # normalize nyr 0-1
      
      # classify catch patterns as Flat, LH, LHL, HL, HLH or OTH
      if(min_max >=0.45 & ct_max.1 >= 0.45 & ct_max.end >= 0.45) { Flat <- 1 } else Flat <- 0
      if(min_max<0.25 & ct_max.1<0.45 & ct_max.end>0.45) { LH <- 1 } else LH <- 0
      if(min_max<0.25 & ct_max.1 < 0.45 & ct_max.end < 0.25) { LHL <- 1 } else LHL <- 0
      if(min_max<0.25 & ct_max.1 > 0.5 & ct_max.end < 0.25) { HL <- 1 } else HL <- 0
      if(min_max<0.25 & ct_max.1 >= 0.45 & ct_max.end >= 0.45) { HLH <- 1 } else HLH <- 0
      if(sum(c(Flat,LHL,LH,HL,HLH))<1) { OTH <- 1 } else OTH <- 0
      
      # Compute predictions for start, end, and int Bk with trained neural networks
      # B/k range that contains 90% of the data points if ct/MSY.pr >= 1
      bk.MSY <- c(0.256 , 0.721 ) # based on all ct/MSY.pr data for 400 stocks # data copied from Plot_ct_MSY_13.R output
      CL.1   <- c( 0.01 , 0.203 )
      CL.2   <- c( 0.2 , 0.431 )
      CL.3   <- c( 0.8 , -0.45 )
      CL.4   <- c( 1.02 , -0.247 )
      
      # estimate startbio
      # if ct/MSY.pr >= ct_MSY.lim use bk.MSY range
      if(mean.ct_MSY.start >= ct_MSY.lim) {
        startbio    <- bk.MSY
      } else { # else run neural network to determine whether B/k is above or below 0.5
        nninput.start  <- as.data.frame(cbind(Flat,LH,LHL,HL,HLH,OTH,min_max,max.ct.i,min.ct.i,yr.norm, #ct_MSY.1,
          mean.ct_MSY.start,slope.first.nrm,mean.ct_MSY.end,slope.last.nrm)) #gm.prior.r
        pr.nn.startbio <- compute(nn.startbio, nninput.start)
        pr.nn_indices.startbio <- max.col(pr.nn.startbio$net.result)
        ct_MSY.use     <- ifelse(ct_MSY.1 < mean.ct_MSY.start,ct_MSY.1,mean.ct_MSY.start)
        if(pr.nn_indices.startbio==1) { # if nn predicts B/k below 0.5
          startbio      <- c(CL.1[1]+CL.1[2]*mean.ct_MSY.start,CL.2[1]+CL.2[2]*mean.ct_MSY.start) } else {
            startbio      <- c(CL.3[1]+CL.3[2]*mean.ct_MSY.start,CL.4[1]+CL.4[2]*mean.ct_MSY.start) }
      } # end of neural network loop
      
      # estimate intbio
      if(ct_MSY.int >= ct_MSY.lim) {
        intbio    <- bk.MSY
      } else { # else run neural network to determine whether B/k is above or below 0.5
        nninput.int    <- as.data.frame(cbind(Flat,LH,LHL,HL,HLH,OTH, # shapes
          min_max,max.ct.i,min.ct.i,yr.norm, # general
          int.ct.i,ct_MSY.int,                   # int
          mean.ct_MSY.end,slope.last.nrm,        # end
          mean.ct_MSY.start,slope.first.nrm))     # start
        
        pr.nn.intbio   <- compute(nn.intbio, nninput.int)
        pr.nn_indices.intbio <- max.col(pr.nn.intbio$net.result)
        if(pr.nn_indices.intbio==1){ # if nn predicts B/k below 0.5
          intbio      <- c(CL.1[1]+CL.1[2]*ct_MSY.int,CL.2[1]+CL.2[2]*ct_MSY.int) } else {
            intbio    <- c(CL.3[1]+CL.3[2]*ct_MSY.int,CL.4[1]+CL.4[2]*ct_MSY.int)}
      } # end of nn loop
      
      # estimate endbio
      # if ct/MSY.pr >= ct_MSY.lim use bk.MSY range
      if(mean.ct_MSY.end >= ct_MSY.lim) {
        endbio    <- bk.MSY
      } else { # else run neural network to determine whether B/k is above or below 0.5
        nninput.end    <- as.data.frame(cbind(Flat,LH,LHL,HL,HLH,OTH,ct_MSY.int,min_max,max.ct.i,  # arbitrary best sequence
          int.ct.i,min.ct.i,yr.norm,
          mean.ct_MSY.start,slope.first.nrm,mean.ct_MSY.end,slope.last.nrm))
        pr.nn.endbio   <- compute(nn.endbio, nninput.end)
        pr.nn_indices.endbio <- max.col(pr.nn.endbio$net.result)
        ct_MSY.use    <- ifelse(ct_MSY.end < mean.ct_MSY.end,ct_MSY.end,mean.ct_MSY.end)
        if(pr.nn_indices.endbio==1){ # if nn predicts B/k below 0.5
          endbio      <- c(CL.1[1]+CL.1[2]*ct_MSY.use,CL.2[1]+CL.2[2]*ct_MSY.use) } else {
            endbio      <- c(CL.3[1]+CL.3[2]*ct_MSY.use,CL.4[1]+CL.4[2]*ct_MSY.use)}
        
      } # end of nn loop
      
      # -------------------------------------------------------
      # if abundance data are available, use to set B/k priors
      #--------------------------------------------------------
      # The following assumes that max smoothed cpue will not exceed carrying capacity and will
      # not be less than a quarter of carrying capacity
      
      if(btype != "None") {
        # get length, min, max, min/max ratio of smoothed bt data
        start.bt      <- yr[which(bt>0)[1]]
        end.bt        <- yr[max(which(bt>0))]
        yr.bt         <- seq(from=start.bt,to=end.bt,by=1) #range of years with bt data
        bt.no.na      <- approx(bt[yr>=start.bt & yr<=end.bt],n=length(yr.bt))$y
        bt.sm         <- ksmooth(x=yr.bt,y=bt.no.na,kernel="normal",n.points=length(yr.bt),bandwidth=bw)$y
        min.bt.sm     <- min(bt.sm,na.rm=T)
        max.bt.sm     <- max(bt.sm,na.rm=T)
        yr.min.bt.sm  <- yr.bt[which.min(bt.sm)]
        yr.max.bt.sm  <- yr.bt[bt.sm==max.bt.sm]
        
        # The prior B/k bounds derived from cpue are Bk.cpue.pr.low = 0.25 * cpue/max.cpue
        # and Bk.cpue.pr.hi = 1.0 * cpue/max.cpue
        if(bt4pr == T) { # if B/k priors shall be estimated from CPUE...
          # if cpue is available in first 3 years, use to set startbio
          if(is.na(stb.low)==T & is.na(stb.hi)==T & start.bt <= yr[3]) {
            startbio.bt <- c(0.25*bt.sm[1]/max.bt.sm,bt.sm[1]/max.bt.sm)
            # if first catch is low and cpue close to max, assume unexploited stock
            if(ct[1]/max.ct < 0.2 & bt.sm[1]/max.bt.sm > 0.8) {startbio.bt <- c(0.8,1)}
            
            # use startbio estimated from bt only if it is narrower or similar to startbio estimated by the neural network
            if((1.25*(startbio[2]-startbio[1])) >  (startbio.bt[2]-startbio.bt[1])) {
              startbio <- startbio.bt }
            
          } # end of startbio loop
          
          # use min cpue to set intbio (ignore years close to start or end)
          if(is.na(intb.low)==T & is.na(intb.hi)==T) {
            st.33     <- ifelse(start.bt<(start.yr+3),start.yr+3,start.bt) # first year eligible for intbio
            end.33    <- ifelse(end.bt>(end.yr-3),end.yr-3,end.bt) # last year eligible for intbio
            bt.33     <- bt.sm[yr.bt>=st.33 & yr.bt<=end.33] # CPUE values relevant for intbio
            yr.bt.33  <- seq(from=st.33,to=end.33,by=1) # range of years with relevant bt data
            min.bt.33 <- min(bt.33,na.rm=T) # mimimum of relevant bt
            int.yr.bt <- yr.bt.33[bt.33==min.bt.33] # year with min bt
            intbio.bt <- c(0.25*min.bt.33/max.bt.sm,min.bt.33/max.bt.sm) # intbio prior predicted for int.yr.bt
            
            # if mean catch/MSY before int.yr is high (> 0.8), use narrower range
            ct.MSY.prev  <- mean(ct[yr>=(int.yr-4) & yr<=int.yr])/MSY.pr
            if(ct.MSY.prev > 0.8) { intbio.bt <- c(1.2*intbio.bt[1],0.8*intbio.bt[2]) }
            # if cpue range is narrow, use lower intbio
            if(min.bt.sm/max.bt.sm>0.3) {intbio.bt <- c(0.8*intbio.bt[1],0.8*intbio.bt[2]) }
            
            # use intbio estimated from bt only if it is narrower or similar to intbio estimated by the neural network
            if((1.25*(intbio[2]-intbio[1])) >=  (intbio.bt[2]-intbio.bt[1])) {
              int.yr   <- int.yr.bt
              intbio   <- intbio.bt }
            
          } # end of intbio loop
          
          # if cpue is within last 3 years of time series, use to set endbio
          if(is.na(endb.low)==T & is.na(endb.hi)==T) {
            if(end.bt >= yr[nyr-2]) {
              endbio.bt  <- c(0.25*bt.sm[yr.bt==end.bt]/max.bt.sm,bt.sm[yr.bt==end.bt]/max.bt.sm)
              # if mean catch/MSY before end.yr is high (> 0.8), use narrower range,
              # because with high previous catch, biomass can neither be very low nor near k
              ct.MSY.prev  <- mean(ct[yr>=(end.yr-4) & yr<=end.yr])/MSY.pr
              if(ct.MSY.prev > 0.8) { endbio.bt <- c(1.2*endbio.bt[1],0.8*endbio.bt[2]) }
              # if endbio estimated by neural network is low and cpue is well below max, use endbio
              if(mean(endbio.bt)>mean(endbio) & mean(endbio)<0.3 & bt.sm[yr.bt==end.bt]/max.bt.sm < 0.7) {endbio.bt <- endbio}
              # if cpue range is narrow, use lower endbio
              if(min.bt.sm/max.bt.sm>0.3) {endbio.bt <- c(0.8*endbio.bt[1],0.8*endbio.bt[2]) }
              
              # use endbio estimated from bt only if it is narrower or similar to endbio estimated by the neural network
              if((1.25*(endbio[2]-endbio[1])) >  (endbio.bt[2]-endbio.bt[1])) {
                endbio   <- endbio.bt }
            }
          } # end of endbio loop
        } # end of b/k prior loop
      } # end of bt priors loop
      
      # if user defined B/k priors in the ID file, use those
      if(is.na(stb.low)==F & is.na(stb.hi)==F) {startbio <- c(stb.low,stb.hi)}
      if(is.na(intb.low)==F & is.na(intb.hi)==F) {
        int.yr   <- cinfo$int.yr[cinfo$Stock==stock]
        intbio   <- c(intb.low,intb.hi)}
      if(is.na(endb.low)==F & is.na(endb.hi)==F) {endbio   <- c(endb.low,endb.hi)}
      
      cat("startbio=",startbio,ifelse(is.na(stb.low)==T,"default","expert"),
        ", intbio=",int.yr,intbio,ifelse(is.na(intb.low)==T,"default","expert"),
        ", endbio=",endbio,ifelse(is.na(endb.low)==T,"default","expert"),"\n")
      
      #----------------------------------------------------------------
      # Multivariate normal sampling of r-k log space
      #----------------------------------------------------------------
      # turn numerical ranges into log-normal distributions
      
      mean.log.r=mean(log(prior.r))
      sd.log.r=(log(prior.r[2])-log(prior.r[1]))/4  # assume range covers 4 SD
      
      mean.log.k <- mean(log(prior.k))
      sd.log.k   <- (log(prior.k[2])-log(prior.k[1]))/4 # assume range covers 4 SD
      
      mvn.log.rk <- mvn(n=n,mean.log.r=mean.log.r,sd.log.r=sd.log.r,mean.log.k=mean.log.k,sd.log.k=sd.log.k)
      #><>MSY rk based on empirical mvn
      ri.emp     <- exp(mvn.log.rk[,1])
      ki1.emp    <- exp(mvn.log.rk[,2])
      
      #-----------------------------------------------------------------
      #Plot data and progress -----
      #-----------------------------------------------------------------
      # check for operating system, open separate window for graphs if Windows
      if(grepl("win",tolower(Sys.info()['sysname']))) {windows(14,9)}
      par(mfrow=c(2,3),mar=c(5.1,4.5,4.1,2.1))
      # (a): plot catch ----
      plot(x=yr, y=ct.raw,
        ylim=c(0,max(ifelse(substr(id_file,1,3)=="Sim",
          1.1*true.MSY,0),1.2*max(ct.raw))),
        type ="l", bty="l", main=paste("A:",gsub(":","",gsub("/","-",stock))), xlab="", ylab="Catch (1000 tonnes/year)", lwd=2, cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
      lines(x=yr,y=ct,col="blue", lwd=1)
      points(x=yr[max.yr.i], y=max.ct, col="red", lwd=2)
      points(x=yr[min.yr.i], y=min.ct, col="red", lwd=2)
      lines(x=yr,y=rep(MSY.pr,length(yr)),lty="dotted",col="purple")
      if(substr(id_file,1,3)=="Sim") lines(x=yr,y=rep(true.MSY,length(yr)),lty="dashed",col="green")
      
      # (b): plot r-k graph
      plot(x=ri1, y=ki1, xlim = c(0.95*quantile(ri1,0.001),1.2*quantile(ri1,0.999)),
        ylim = c(0.95*quantile(ki1,0.001),1.2*quantile(ki1,0.999)),
        log="xy", xlab="r", ylab="k (1000 tonnes)", main="B: Finding viable r-k", pch=".", cex=2, bty="l",
        col=grey(0.7,0.4), cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
      lines(x=c(prior.r[1],prior.r[2],prior.r[2],prior.r[1],prior.r[1]), # plot original prior range
        y=c(prior.k[1],prior.k[1],prior.k[2],prior.k[2],prior.k[1]),
        lty="dotted")
      
      #---------------------------------------------------------------------
      # Prepare MCMC analyses
      #---------------------------------------------------------------------
      # set inits for r-k in lower right corner of log r-k space to avoid intermediate maxima
      init.r      <- prior.r[1]+0.8*(prior.r[2]-prior.r[1])
      init.k      <- prior.k[1]+0.1*(prior.k[2]-prior.k[1])
      
      # vector with no penalty (=0) if predicted biomass is within viable range, else a penalty of 10 is set
      pen.bk = pen.F = rep(0,length(ct))
      
      # Add biomass priors
      b.yrs = c(1,length(start.yr:int.yr),length(start.yr:end.yr))
      b.prior = rbind(matrix(c(startbio[1],startbio[2],intbio[1],intbio[2],endbio[1],endbio[2]),2,3),rep(0,3)) # last row includes the 0 penalty
      
      #----------------------------------------------------------------
      # First run of BSM with only catch data = CMSY++
      #----------------------------------------------------------------
      # changes by RF to account for asymmetric distributions
      bt.start  <- mean(c(prior.k[1]*startbio[1],prior.k[2]*startbio[2])) # derive proxy for first bt value
      bt.cmsy   <- c(bt.start,rep(NA,length(ct)-1)) # create proxy abundance with one start value and rest = NA
      bt.int    <- mean(c(prior.k[1]*intbio[1],prior.k[2]*intbio[2]))
      bt.last  <- mean(c(prior.k[1]*endbio[1],prior.k[2]*endbio[2]))
      
      mean.cmsy.ct   <- mean(c(ct[1],ct[yr==int.yr],ct[nyr]),na.rm=T) # get mean catch of years with prior bt
      mean.cmsy.cpue <- mean(c(bt.start,bt.int,bt.last),na.rm=T) # get mean of prior bt
      
      q.prior.cmsy    <- c(0.99,1.01) # since no abundance data are available in this run,
      init.q.cmsy     <- 1            # q could be omitted and is set here to (practically) 1
      
      cat("Running MCMC analysis with only catch data....\n")
      
      # call Schaefer model function
      jags_cmsy <- bsm(ct=ct,btj=bt.cmsy,nyr=nyr,prior.r=prior.r,prior.k=prior.k,startbio=startbio,q.priorj=q.prior.cmsy,
        init.q=init.q.cmsy,init.r=init.r,init.k=init.k,pen.bk=pen.bk,pen.F=pen.F,b.yrs=b.yrs,
        b.prior=b.prior,CV.C=CV.C,CV.cpue=CV.cpue,nbk=nbk,rk.cor.beta=rk.cor.beta,cmsyjags=TRUE)
      
      #-----------------------------------------------
      # Get CMSY++ results
      #-----------------------------------------------
      rs                <- as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$r))   # unique.rk[,1]
      ks                <- as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$k))  # unique.rk[,2]
      ellipse.cmsy       <- traceEllipse(rs,ks,prior.r,prior.k) # GP
      r.cmsy             <- ellipse.cmsy[1] # GP
      k.cmsy             <- ellipse.cmsy[2] # GP
      # restrict CI quantiles to above 25th percentile of rs
      rs.025             <- as.numeric(quantile(rs,0.025))
      r.quant.cmsy       <- as.numeric(quantile(rs[rs>rs.025],c(0.5,0.025,0.975))) # median, 95% CIs in range around
      k.quant.cmsy       <- as.numeric(quantile(ks[rs>rs.025],c(0.5,0.025,0.975)))
      lcl.r.cmsy         <- r.quant.cmsy[2]
      ucl.r.cmsy         <- r.quant.cmsy[3]
      lcl.k.cmsy         <- k.quant.cmsy[2]
      ucl.k.cmsy         <- k.quant.cmsy[3]
      MSY.quant.cmsy     <- quantile(rs[rs>rs.025]*ks[rs>rs.025]/4,c(0.5,0.025,0.975))
      MSY.cmsy           <- r.cmsy*k.cmsy/4
      lcl.MSY.cmsy       <- MSY.quant.cmsy[2]
      ucl.MSY.cmsy       <- MSY.quant.cmsy[3]
      qs                 <- as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$q))
      q.quant.cmsy       <- quantile(qs,c(0.5,0.025,0.975))
      q.cmsy             <- q.quant.cmsy[1]
      lcl.q.cmsy         <- q.quant.cmsy[2]
      ucl.q.cmsy         <- q.quant.cmsy[3]
      
      Fmsy.quant.cmsy    <- as.numeric(quantile(rs[rs>rs.025]/2,c(0.5,0.025,0.975)))
      Fmsy.cmsy          <- r.cmsy/2 # HW checked
      lcl.Fmsy.cmsy      <- Fmsy.quant.cmsy[2] #><>HW to be added to report output
      ucl.Fmsy.cmsy      <- Fmsy.quant.cmsy[3] #><>HW to be added to report output
      Bmsy.quant.cmsy    <- as.numeric(quantile(ks[rs>rs.025]/2,c(0.5,0.025,0.975)))
      Bmsy.cmsy          <- k.cmsy/2 # HW checked
      lcl.Bmsy.cmsy      <- Bmsy.quant.cmsy[2] #><>HW to be added to report output
      ucl.Bmsy.cmsy      <- Bmsy.quant.cmsy[3] #><>HW to be added to report output
      # HW posterior predictives can stay unchanged
      ppd.r              <- exp(as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$ppd.logrk[,1])))
      ppd.k              <- exp(as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$ppd.logrk[,2])))
      
      #><>HW get FFmsy directly from JAGS
      all.FFmsy.cmsy  = jags_cmsy$BUGSoutput$sims.list$FFmsy
      FFmsy.quant.cmsy = apply(all.FFmsy.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
      FFmsy.cmsy = FFmsy.quant.cmsy[1,]
      lcl.FFmsy.cmsy = FFmsy.quant.cmsy[2,]
      ucl.FFmsy.cmsy = FFmsy.quant.cmsy[3,]
      #><>HW get BBmsy directly from JAGS
      all.BBmsy.cmsy  = jags_cmsy$BUGSoutput$sims.list$BBmsy
      BBmsy.quant.cmsy = apply(all.BBmsy.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
      BBmsy.cmsy = BBmsy.quant.cmsy[1,]
      lcl.BBmsy.cmsy = BBmsy.quant.cmsy[2,]
      ucl.BBmsy.cmsy = BBmsy.quant.cmsy[3,]
      # get relative biomass P=B/k as predicted by BSM, including predictions for years with NA abundance
      all.bk.cmsy  = jags_cmsy$BUGSoutput$sims.list$P
      bk.quant.cmsy = apply(all.bk.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
      bk.cmsy = bk.quant.cmsy[1,]
      lcl.bk.cmsy = bk.quant.cmsy[2,]
      ucl.bk.cmsy = bk.quant.cmsy[3,]
      #><> NEW get biomass from JAGS posterior
      all.B.cmsy  = jags_cmsy$BUGSoutput$sims.list$B
      B.quant.cmsy = apply(all.B.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
      B.cmsy = B.quant.cmsy[1,]
      lcl.B.cmsy = B.quant.cmsy[2,]
      ucl.B.cmsy = B.quant.cmsy[3,]
      #><> NEW get F from JAGS posterior
      all.Ft.cmsy  = jags_cmsy$BUGSoutput$sims.list$F
      Ft.quant.cmsy = apply(all.Ft.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
      Ft.cmsy = Ft.quant.cmsy[1,]
      lcl.Ft.cmsy = Ft.quant.cmsy[2,]
      ucl.Ft.cmsy = Ft.quant.cmsy[3,]
      
      # get catch estimates given catch CV
      all.ct.cmsy  = jags_cmsy$BUGSoutput$sims.list$ct.jags
      ct.quants.cmsy = apply(all.ct.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
      ct.cmsy          <- ct.quants.cmsy[1,]
      lcl.ct.cmsy      <- ct.quants.cmsy[2,]
      ucl.ct.cmsy      <- ct.quants.cmsy[3,]
      
      #-------------------------------------------------------------------
      # Plot results
      #-------------------------------------------------------------------
      # (b) continued
      # plot viable r-k pairs from catch-only BSM run
      points(x=rs,y=ks,pch=".",cex=1,col="gray55")
      
      # show CMSY++ estimate in prior space of graph B
      points(x=r.cmsy, y=k.cmsy, pch=19, col="blue")
      lines(x=c(lcl.r.cmsy, ucl.r.cmsy),y=c(k.cmsy,k.cmsy), col="blue")
      lines(x=c(r.cmsy,r.cmsy),y=c(lcl.k.cmsy, ucl.k.cmsy), col="blue")
      
      lines(x=c(prior.r[1],prior.r[2],prior.r[2],prior.r[1],prior.r[1]), # re-plot original prior range
        y=c(prior.k[1],prior.k[1],prior.k[2],prior.k[2],prior.k[1]),lty="dotted")
      
      # ------------------------------------------------------------------
      # Second run with Bayesian analysis of catch & biomass (or CPUE) with Schaefer model ----
      # ------------------------------------------------------------------
      FullSchaefer <- F
      #   bt           <- bt.raw
      if(btype != "None" & length(bt[is.na(bt)==F])>=nab) {
        FullSchaefer <- T
        cat("Running MCMC analysis with catch and CPUE.... \n")
        
        if(btype=="biomass") {
          q.prior <- q.biomass.pr
          init.q  <- mean(q.prior)
        } else { # if btype is CPUE
          # get mean of 3 highest bt values
          bt.sort <- sort(bt)
          mean.max.bt <- mean(bt.sort[(length(bt.sort)-2):length(bt.sort)],na.rm = T)
          # Estimate q.prior[2] from max cpue = q * k, q.prior[1] from max cpue = q * 0.25 * k
          q.1           <- mean.max.bt/prior.k[2]
          q.2           <- mean.max.bt/(0.25*prior.k[1])
          q.prior       <- c(q.1,q.2)
          q.init        <- mean(q.prior) }
        
        # call Schaefer model function
        jags_bsm <- bsm(ct=ct,btj=bt,nyr=nyr,prior.r=prior.r,prior.k=prior.k,startbio=startbio,q.priorj=q.prior,
          init.q=init.q,init.r=init.r,init.k=init.k,pen.bk=pen.bk,pen.F=pen.F,b.yrs=b.yrs,
          b.prior=b.prior,CV.C=CV.C,CV.cpue=CV.cpue,nbk=nbk,rk.cor.beta=rk.cor.beta,cmsyjags=FALSE)
        
        # --------------------------------------------------------------
        # Results from BSM Schaefer - ><>HW now consistent with CMSY++
        # --------------------------------------------------------------
        rs.bsm            <- as.numeric(mcmc(jags_bsm$BUGSoutput$sims.list$r))   # unique.rk[,1]
        ks.bsm            <- as.numeric(mcmc(jags_bsm$BUGSoutput$sims.list$k))  # unique.rk[,2]
        #><> HW: Go directly with posterior median and CIs (non-parametric)
        r.quant.bsm       <- as.numeric(quantile(rs.bsm,c(0.5,0.025,0.975))) #median, 95% CIs
        r.bsm             <- r.quant.bsm[1]
        lcl.r.bsm         <- r.quant.bsm[2]
        ucl.r.bsm         <- r.quant.bsm[3]
        k.quant.bsm          <- as.numeric(quantile(ks.bsm,c(0.5,0.025,0.975)))
        k.bsm             <- k.quant.bsm[1]
        lcl.k.bsm         <- k.quant.bsm[2]
        ucl.k.bsm         <- k.quant.bsm[3]
        MSY.quant.bsm     <- quantile(rs.bsm*ks.bsm/4,c(0.5,0.025,0.975))
        MSY.bsm           <- MSY.quant.bsm[1]
        lcl.MSY.bsm       <- MSY.quant.bsm[2]
        ucl.MSY.bsm       <- MSY.quant.bsm[3]
        qs.bsm            <- as.numeric(mcmc(jags_bsm$BUGSoutput$sims.list$q))
        q.quant.bsm       <- as.numeric(quantile(qs.bsm,c(0.5,0.025,0.975)))
        q.bsm             <- q.quant.bsm[1]
        lcl.q.bsm         <- q.quant.bsm[2]
        ucl.q.bsm         <- q.quant.bsm[3]
        
        Fmsy.quant.bsm      <- as.numeric(quantile(rs.bsm/2,c(0.5,0.025,0.975)))
        Fmsy.bsm           <- Fmsy.quant.bsm[1]
        lcl.Fmsy.bsm        <- Fmsy.quant.bsm[2] #><>HW to be added to report output
        ucl.Fmsy.bsm        <- Fmsy.quant.bsm[3] #><>HW to be added to report output
        Bmsy.quant.bsm      <- as.numeric(quantile(ks.bsm/2,c(0.5,0.025,0.975)))
        Bmsy.bsm            <- Bmsy.quant.bsm[1]
        lcl.Bmsy.bsm        <- Bmsy.quant.bsm[2] #><>HW to be added to report output
        ucl.Bmsy.bsm        <- Bmsy.quant.bsm[3] #><>HW to be added to report output
        
        #><>HW get FFmsy directly from JAGS
        all.FFmsy.bsm  = jags_bsm$BUGSoutput$sims.list$FFmsy
        FFmsy.quant.bsm = apply(all.FFmsy.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
        FFmsy.bsm = FFmsy.quant.bsm[1,]
        lcl.FFmsy.bsm = FFmsy.quant.bsm[2,]
        ucl.FFmsy.bsm = FFmsy.quant.bsm[3,]
        #><>HW get BBmsy directly from JAGS
        all.BBmsy.bsm  = jags_bsm$BUGSoutput$sims.list$BBmsy
        BBmsy.quant.bsm = apply(all.BBmsy.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
        BBmsy.bsm = BBmsy.quant.bsm[1,]
        lcl.BBmsy.bsm = BBmsy.quant.bsm[2,]
        ucl.BBmsy.bsm = BBmsy.quant.bsm[3,]
        # get relative biomass P=B/k as predicted by BSM, including predictions for years with NA abundance
        all.bk.bsm  = jags_bsm$BUGSoutput$sims.list$P
        bk.quant.bsm = apply(all.bk.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
        bk.bsm = bk.quant.bsm[1,]
        lcl.bk.bsm = bk.quant.bsm[2,]
        ucl.bk.bsm = bk.quant.bsm[3,]
        #><> NEW get biomass from JAGS posterior
        all.B.bsm  = jags_bsm$BUGSoutput$sims.list$B
        B.quant.bsm = apply(all.B.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
        B.bsm = B.quant.bsm[1,]
        lcl.B.bsm = B.quant.bsm[2,]
        ucl.B.bsm = B.quant.bsm[3,]
        #><> NEW get F from JAGS posterior
        all.Ft.bsm  = jags_bsm$BUGSoutput$sims.list$F
        Ft.quant.bsm = apply(all.Ft.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
        Ft.bsm = Ft.quant.bsm[1,]
        lcl.Ft.bsm = Ft.quant.bsm[2,]
        ucl.Ft.bsm = Ft.quant.bsm[3,]
        
        # get catch estimates given catch CV
        all.ct.bsm  = jags_bsm$BUGSoutput$sims.list$ct.jags
        ct.quants.bsm = apply(all.ct.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
        ct.bsm          <- ct.quants.bsm[1,]
        lcl.ct.bsm      <- ct.quants.bsm[2,]
        ucl.ct.bsm      <- ct.quants.bsm[3,]
        
        #-------------------------------------------
        # BSM fits
        #-------------------------------------------
        #><> HW PLOT E (observations)
        F.bt.jags       <- q.bsm*ct.raw/bt # F from raw data
        F.bt_Fmsy.jags  <- vector() # initialize vector
        for(z in 1: length(F.bt.jags)) {
          F.bt_Fmsy.jags[z] <- ifelse(is.na(bt[z])==T,NA,F.bt.jags[z]/
              ifelse(((bt[z]/q.bsm)/k.bsm)<0.25,Fmsy.bsm*4*(bt[z]/q.bsm)/k.bsm,Fmsy.bsm))}
        
        #><> get cpue fits from BSM
        cpue.bsm        <- exp(jags_bsm$BUGSoutput$sims.list$cpuem)
        pe.logbt.bsm   <- (jags_bsm$BUGSoutput$sims.list$proc.logB)
        # get cpue predicted
        pred.cpue            <- apply(cpue.bsm,2,quantile,c(0.5,0.025,0.975))
        cpue.bsm          <- pred.cpue[1,]
        lcl.cpue.bsm      <- pred.cpue[2,]
        ucl.cpue.bsm      <- pred.cpue[3,]
        # get process error on log(biomass)   pred.cpue            <- apply(cpue.jags,2,quantile,c(0.5,0.025,0.975))
        pred.pe         <- apply(pe.logbt.bsm,2,quantile,c(0.5,0.025,0.975))
        pe.bsm         <- pred.pe[1,]
        lcl.pe.bsm     <- pred.pe[2,]
        ucl.pe.bsm     <- pred.pe[3,]
        
        
        # get variance and correlation between log(r) and log(k)
        log.r.var    <- var(x=log(rs.bsm))
        log.k.var    <- var(x=log(ks.bsm))
        log.kr.cor   <- cor(x=log(rs.bsm),y=log(ks.bsm))
        log.kr.cov   <- cov(x=log(rs.bsm),y=log(ks.bsm))
        
      } # end of MCMC BSM Schaefer loop
      
      # --------------------------------------------
      # Get results for management ----
      # --------------------------------------------
      if(FullSchaefer==F | force.cmsy==T) { # if only CMSY is available or shall be used
        MSY   <-MSY.cmsy; lcl.MSY<-lcl.MSY.cmsy; ucl.MSY<-ucl.MSY.cmsy
        Bmsy  <-Bmsy.cmsy; lcl.Bmsy<-lcl.Bmsy.cmsy; ucl.Bmsy<-ucl.Bmsy.cmsy
        Fmsy  <-Fmsy.cmsy; lcl.Fmsy<-lcl.Fmsy.cmsy; ucl.Fmsy<-ucl.Fmsy.cmsy
        F.Fmsy<-FFmsy.cmsy;lcl.F.Fmsy<-lcl.FFmsy.cmsy; ucl.F.Fmsy<-ucl.FFmsy.cmsy
        B.Bmsy<-BBmsy.cmsy[1:nyr];lcl.B.Bmsy<-lcl.BBmsy.cmsy[1:nyr][1:nyr];ucl.B.Bmsy<-ucl.BBmsy.cmsy[1:nyr]
        B <- B.cmsy[1:nyr];lcl.B<-lcl.B.cmsy[1:nyr][1:nyr];ucl.B<-ucl.B.cmsy[1:nyr]
        Ft <- Ft.cmsy[1:nyr];lcl.Ft<-lcl.Ft.cmsy[1:nyr][1:nyr];ucl.Ft<-ucl.Ft.cmsy[1:nyr]
        bk <- bk.cmsy[1:nyr];lcl.bk<-lcl.bk.cmsy[1:nyr][1:nyr];ucl.bk<-ucl.bk.cmsy[1:nyr]
        
        ct.jags <- ct.cmsy; lcl.ct.jags = lcl.ct.cmsy; ucl.ct.jags=ucl.ct.cmsy #catch estimate given catch error
        
      } else { # if FullSchaefer is TRUE
        MSY   <-MSY.bsm; lcl.MSY<-lcl.MSY.bsm; ucl.MSY<-ucl.MSY.bsm
        Bmsy  <-Bmsy.bsm; lcl.Bmsy<-lcl.Bmsy.bsm; ucl.Bmsy<-ucl.Bmsy.bsm
        Fmsy  <-Fmsy.bsm; lcl.Fmsy<-lcl.Fmsy.bsm; ucl.Fmsy<-ucl.Fmsy.bsm
        F.Fmsy<-FFmsy.bsm;lcl.F.Fmsy<-lcl.FFmsy.bsm; ucl.F.Fmsy<-ucl.FFmsy.bsm
        B.Bmsy<-BBmsy.bsm[1:nyr];lcl.B.Bmsy<-lcl.BBmsy.bsm[1:nyr][1:nyr];ucl.B.Bmsy<-ucl.BBmsy.bsm[1:nyr]
        B <- B.bsm[1:nyr];lcl.B<-lcl.B.bsm[1:nyr][1:nyr];ucl.B<-ucl.B.bsm[1:nyr]
        Ft <- Ft.bsm[1:nyr];lcl.Ft<-lcl.Ft.bsm[1:nyr][1:nyr];ucl.Ft<-ucl.Ft.bsm[1:nyr]
        bk <- bk.bsm[1:nyr];lcl.bk<-lcl.bk.bsm[1:nyr][1:nyr];ucl.bk<-ucl.bk.bsm[1:nyr]
        ct.jags <- ct.bsm; lcl.ct.jags = lcl.ct.bsm; ucl.ct.jags=ucl.ct.bsm #catch estimate given catch error
        
      }
      
      #><> New section simplified for CMSY++ and BSM
      Fmsy.adj     <- ifelse(B.Bmsy>0.5,Fmsy,Fmsy*2*B.Bmsy)
      lcl.Fmsy.adj <- ifelse(B.Bmsy>0.5,lcl.Fmsy,lcl.Fmsy*2*B.Bmsy)
      ucl.Fmsy.adj <- ifelse(B.Bmsy>0.5,ucl.Fmsy,ucl.Fmsy*2*B.Bmsy)
      
      if(is.na(sel.yr)==F){
        B.Bmsy.sel<-B.Bmsy[yr==sel.yr]
        B.sel<-B.Bmsy.sel*Bmsy
        F.sel<-ct.raw[yr==sel.yr]/B.sel
        F.Fmsy.sel<-F.sel/Fmsy.adj[yr==sel.yr]
      }
      
      # ------------------------------------------
      # print input and results to screen ----
      #-------------------------------------------
      cat("---------------------------------------\n")
      cat("Species:", cinfo$ScientificName[cinfo$Stock==stock], ", stock:",stock,", ",ename,"\n")
      cat(cinfo$Name[cinfo$Stock==stock], "\n")
      cat("Region:",cinfo$Region[cinfo$Stock==stock],",",cinfo$Subregion[cinfo$Stock==stock],"\n")
      cat("Catch data used from years", min(yr),"-", max(yr),", abundance =", btype, "\n")
      cat("Prior initial relative biomass =", startbio[1], "-", startbio[2],ifelse(is.na(stb.low)==T,"default","expert"), "\n")
      cat("Prior intermediate rel. biomass=", intbio[1], "-", intbio[2], "in year", int.yr,ifelse(is.na(intb.low)==T,"default","expert"), "\n")
      cat("Prior final relative biomass   =", endbio[1], "-", endbio[2],ifelse(is.na(endb.low)==T,"default","expert"), "\n")
      cat("Prior range for r =", format(prior.r[1],digits=2), "-", format(prior.r[2],digits=2),ifelse(is.na(r.low)==T,"default","expert"),
        ", prior range for k =", prior.k[1], "-", prior.k[2],", MSY prior =",MSY.pr,"\n")
      # if Schaefer and CPUE, print prior range of q
      if(FullSchaefer==T) {
        cat("B/k prior used for first year in BSM",ifelse(nbk>1,"and intermediate year",""),ifelse(nbk==3,"and last year",""),"\n")
        cat("Prior range of q =",q.prior[1],"-",q.prior[2],", assumed effort creep",e.creep,"%\n") }
      if(substr(id_file,1,3)=="Sim") { # if data are simulated, print true values
        cat("True values: r =",true.r,", k = 1000, MSY =", true.MSY,", last B/k =", true.Bk,
          ", last F/Fmsy =",true.F_Fmsy,", q = 0.01\n") }
      
      # results of CMSY analysis
      cat("\nResults of CMSY analysis \n")
      cat("-------------------------\n")
      cat("r   =", r.cmsy,", 95% CL =", lcl.r.cmsy, "-", ucl.r.cmsy,", k =", k.cmsy,", 95% CL =", lcl.k.cmsy, "-", ucl.k.cmsy,"\n")
      cat("MSY =", MSY.cmsy,", 95% CL =", lcl.MSY.cmsy, "-", ucl.MSY.cmsy,"\n")
      cat("Relative biomass in last year =", bk.cmsy[nyr], "k, 2.5th perc =", lcl.bk.cmsy[nyr],
        ", 97.5th perc =", ucl.bk.cmsy[nyr],"\n")
      cat("Exploitation F/(r/2) in last year =", FFmsy.cmsy[nyr],", 2.5th perc =",lcl.FFmsy.cmsy[nyr],
        ", 97.5th perc =",ucl.FFmsy.cmsy[nyr],"\n\n")
      
      
      # print results from full Schaefer if available
      if(FullSchaefer==T) {
        cat("Results from Bayesian Schaefer model (BSM) using catch &",btype,"\n")
        cat("------------------------------------------------------------\n")
        cat("q   =", q.bsm,", lcl =", lcl.q.bsm, ", ucl =", ucl.q.bsm,"(derived from catch and CPUE) \n")
        cat("r   =", r.bsm,", 95% CL =", lcl.r.bsm, "-", ucl.r.bsm,", k =", k.bsm,", 95% CL =", lcl.k.bsm, "-", ucl.k.bsm,", r-k log correlation =", log.kr.cor,"\n")
        cat("MSY =", MSY.bsm,", 95% CL =", lcl.MSY.bsm, "-", ucl.MSY.bsm,"\n")
        cat("Relative biomass in last year =", bk.bsm[nyr], "k, 2.5th perc =",lcl.bk.bsm[nyr],
          ", 97.5th perc =", ucl.bk.bsm[nyr],"\n")
        cat("Exploitation F/(r/2) in last year =", FFmsy.bsm[nyr],", 2.5th perc =",lcl.FFmsy.bsm[nyr],
          ", 97.5th perc =",ucl.FFmsy.bsm[nyr],"\n\n")
      }
      
      # print results to be used in management
      cat("Results for Management (based on",ifelse(FullSchaefer==F | force.cmsy==T,"CMSY","BSM"),"analysis) \n")
      cat("-------------------------------------------------------------\n")
      if(force.cmsy==T) cat("Mangement results based on CMSY because abundance data seem unrealistic\n")
      cat("Fmsy =",Fmsy,", 95% CL =",lcl.Fmsy,"-",ucl.Fmsy,"(if B > 1/2 Bmsy then Fmsy = 0.5 r)\n")
      cat("Fmsy =",Fmsy.adj[nyr],", 95% CL =",lcl.Fmsy.adj[nyr],"-",ucl.Fmsy.adj[nyr],"(r and Fmsy are linearly reduced if B < 1/2 Bmsy)\n")
      cat("MSY  =",MSY,", 95% CL =",lcl.MSY,"-",ucl.MSY,"\n")
      cat("Bmsy =",Bmsy,", 95% CL =",lcl.Bmsy,"-",ucl.Bmsy,"\n")
      cat("Biomass in last year =",B[nyr],", 2.5th perc =", lcl.B[nyr], ", 97.5 perc =",ucl.B[nyr],"\n")
      cat("B/Bmsy in last year  =",B.Bmsy[nyr],", 2.5th perc =", lcl.B.Bmsy[nyr], ", 97.5 perc =",ucl.B.Bmsy[nyr],"\n")
      cat("Fishing mortality in last year =",Ft[nyr],", 2.5th perc =", lcl.Ft[nyr], ", 97.5 perc =",ucl.Ft[nyr],"\n")
      cat("Exploitation F/Fmsy  =",F.Fmsy[nyr],", 2.5th perc =", lcl.F.Fmsy[nyr], ", 97.5 perc =",ucl.F.Fmsy[nyr],"\n")
      
      # show stock status and exploitation for optional selected year
      if(is.na(sel.yr)==F) {
        cat("\nStock status and exploitation in",sel.yr,"\n")
        cat("Biomass =",B.sel, ", B/Bmsy =",B.Bmsy.sel,", F =",F.sel,", F/Fmsy =",F.Fmsy.sel,"\n") }
      
      cat("Comment:", comment,"\n")
      cat("----------------------------------------------------------\n")
      
      # -----------------------------------------
      # Plot results ----
      # -----------------------------------------
      # (b) continued
      # plot best r-k from full Schaefer analysis in prior space of graph B
      if(FullSchaefer==T) {
        points(x=r.bsm, y=k.bsm, pch=19, col="red")
        lines(x=c(lcl.r.bsm, ucl.r.bsm),y=c(k.bsm,k.bsm), col="red")
        lines(x=c(r.bsm,r.bsm),y=c(lcl.k.bsm, ucl.k.bsm), col="red")
      }
      if(substr(id_file,1,3)=="Sim") points(x=true.r,y=true.k,col="green", cex=3, lwd=2)
      
      
      # (c) Analysis of viable r-k plot -----
      # ----------------------------
      max.y    <- max(c(ifelse(FullSchaefer==T,max(ks.bsm,ucl.k.bsm),NA),
        ifelse(substr(id_file,1,3)=="Sim",1.2*true.k,NA),ks),na.rm=T)
      min.y    <- min(c(ifelse(FullSchaefer==T,min(ks.bsm),NA),ks,
        ifelse(substr(id_file,1,3)=="Sim",0.8*true.k,NA)),na.rm=T)
      max.x    <- max(c(ifelse(FullSchaefer==T,max(rs.bsm),NA),rs),na.rm=T)
      min.x    <- min(c(ifelse(FullSchaefer==T,min(rs.bsm),NA),0.9*lcl.r.cmsy,prior.r[1],rs),na.rm=T)
      
      plot(x=rs, y=ks, xlim=c(min.x,max.x),
        ylim=c(min.y,max.y),
        pch=16, col="gray",log="xy", bty="l",
        xlab="", ylab="k (1000 tonnes)", main="C: Analysis of viable r-k",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
      title(xlab = "r", line = 2.25, cex.lab = 1.55)
      
      # plot r-k pairs from MCMC
      if(FullSchaefer==T) {points(x=rs.bsm, y=ks.bsm, pch=16,cex=0.5)}
      
      # plot best r-k from full Schaefer analysis
      if(FullSchaefer==T) {
        points(x=r.bsm, y=k.bsm, pch=19, col="red")
        lines(x=c(lcl.r.bsm, ucl.r.bsm),y=c(k.bsm,k.bsm), col="red")
        lines(x=c(r.bsm,r.bsm),y=c(lcl.k.bsm, ucl.k.bsm), col="red")
      }
      
      # plot blue dot for CMSY r-k, with 95% CL lines
      points(x=r.cmsy, y=k.cmsy, pch=19, col="blue")
      lines(x=c(lcl.r.cmsy, ucl.r.cmsy),y=c(k.cmsy,k.cmsy), col="blue")
      lines(x=c(r.cmsy,r.cmsy),y=c(lcl.k.cmsy, ucl.k.cmsy), col="blue")
      
      if(substr(id_file,1,3)=="Sim") points(x=true.r,y=true.k,col="green", cex=3, lwd=2)
      
      # (d) Pred. biomass plot ----
      #--------------------
      # determine k to use for red line in b/k plot
      if(FullSchaefer==T)  {k2use <- k.bsm} else {k2use <- k.cmsy}
      # determine hight of y-axis in plot
      max.y  <- max(c(ucl.bk.cmsy,ifelse(FullSchaefer==T,max(ucl.bk.bsm[1:nyr]),NA),
        ifelse(FullSchaefer==T,max(bt/(q.bsm*k.bsm),na.rm=T),NA),
        0.6,startbio[2],endbio[2],intbio[2]),na.rm=T)
      max.y  <- ifelse(max.y>4,4,max.y)
      # Main plot of relative CMSY biomass
      plot(x=yr,y=bk.cmsy[1:nyr], lwd=1.5, xlab="", ylab="Relative biomass B/k", type="l",
        ylim=c(0,max.y), bty="l", main="D: Stock size",col="blue",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
      lines(x=yr, y=lcl.bk.cmsy[1:nyr],type="l",lty="dotted",col="blue")
      lines(x=yr, y=ucl.bk.cmsy[1:nyr],type="l",lty="dotted",col="blue")
      # plot lines for 0.5 and 0.25 biomass
      abline(h=0.5, lty="dashed")
      abline(h=0.25, lty="dotted")
      # Add BSM
      if(FullSchaefer==T){
        lines(x=yr, y=bk.bsm[1:nyr],type="l",col="red")
        lines(x=yr, y=lcl.bk.bsm[1:nyr],type="l",lty="dotted",col="red")
        lines(x=yr, y=ucl.bk.bsm[1:nyr],type="l",lty="dotted",col="red")
        # Add CPUE points
        points(x=yr,y=bt/(q.bsm*k.bsm),pch=21,bg="grey")
      }
      # plot biomass windows
      lines(x=c(yr[1],yr[1]), y=startbio, col="purple",lty=ifelse(is.na(stb.low)==T,"dotted","solid"))
      lines(x=c(int.yr,int.yr), y=intbio, col="purple",lty=ifelse(is.na(intb.low)==T,"dotted","solid"))
      lines(x=c(max(yr),max(yr)), y=endbio, col="purple",lty=ifelse(is.na(endb.low)==T,"dotted","solid"))
      
      # if CPUE has been corrected for effort creep, display uncorrected CPUE
      if(btype=="CPUE" & FullSchaefer==T & e.creep.line==T & is.na(e.creep)==FALSE) {
        lines(x=yr,y=bt.raw/(q.bsm*k.bsm),type="l", col="green", lwd=1)
      }
      if(substr(id_file,1,3)=="Sim") points(x=yr[nyr],y=true.Bk,col="green", cex=3, lwd=2)
      
      # (e) Exploitation rate plot ----
      # -------------------------
      # if CPUE data are available but fewer than nab years, plot on second axis
      if(btype == "CPUE" | btype=="biomass") {
        q=1/(max(bk.cmsy[1:nyr][is.na(bt)==F],na.rm=T)*k.cmsy/max(bt,na.rm=T))
        u.cpue      <- q.bsm*ct/bt
      }
      # determine upper bound of Y-axis
      max.y <- max(c(1.5,ucl.FFmsy.cmsy,ifelse(FullSchaefer==T,max(c(ucl.FFmsy.bsm),na.rm=T),NA),na.rm=T),na.rm=T)
      max.y <- ifelse(max.y>10,10,max.y)
      # plot F from CMSY
      plot(x=yr,y=FFmsy.cmsy, type="l", bty="l", lwd=1.5, ylim=c(0,max.y), xlab="",
        ylab=expression(F/F[MSY]), main="E: Exploitation rate", col="blue",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
      lines(x=yr,y=lcl.FFmsy.cmsy,lty="dotted",col="blue")
      lines(x=yr,y=ucl.FFmsy.cmsy,lty="dotted",col="blue")
      abline(h=1, lty="dashed")
      
      # plot F/Fmsy as points from observed catch and CPUE and as red curves from BSM predicted catch and biomass
      if(FullSchaefer==T){
        points(x=yr, y=F.bt_Fmsy.jags, pch=21,bg="grey")
        lines(x=yr,y=FFmsy.bsm, col="red")
        lines(x=yr,y=lcl.FFmsy.bsm, col="red",lty="dotted")
        lines(x=yr,y=ucl.FFmsy.bsm, col="red",lty="dotted")
      }
      if(substr(id_file,1,3)=="Sim") points(x=yr[nyr],y=true.F_Fmsy,col="green", cex=3, lwd=2)
      
      # (f) Parabola plot ----
      #-------------------------
      max.y <- max(c(ct/MSY.cmsy,ifelse(FullSchaefer==T,max(ct/MSY.bsm),NA),1.2),na.rm=T)
      # plot parabola
      x=seq(from=0,to=2,by=0.001)
      y.c  <- ifelse(x>0.25,1,ifelse(x>0.125,4*x,exp(-10*(0.125-x))*4*x)) # correction for low recruitment below half and below quarter of Bmsy
      y=(4*x-(2*x)^2)*y.c
      plot(x=x, y=y, xlim=c(0,1), ylim=c(0,max.y), type="l", bty="l",xlab="",
        ylab="Catch / MSY", main="F: Equilibrium curve",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
      title(xlab= "Relative biomass B/k", line = 2.25, cex.lab = 1.55)
      
      # plot catch against CMSY estimates of relative biomass
      #><> HW add catch with error from JAGS
      lines(x=bk.cmsy[1:nyr], y=ct.cmsy/MSY.cmsy, pch=16, col="blue", lwd=1)
      points(x=bk.cmsy[1], y=ct.cmsy[1]/MSY.cmsy[1], pch=0, cex=2, col="blue")
      points(x=bk.cmsy[nyr], y=ct.cmsy[length(ct)]/MSY.cmsy[length(MSY.cmsy)],cex=2,pch=2,col="blue")
      
      # for CPUE, plot catch scaled by BSM MSY against observed biomass derived as q * CPUE scaled by BSM k
      if(FullSchaefer==T) {
        points(x=bt/(q.bsm*k.bsm), y=ct/MSY.bsm, pch=21,bg="grey")
        lines(x=bk.bsm[1:nyr], y=ct.bsm/MSY.bsm, pch=16, col="red",lwd=1)
        points(x=bk.bsm[1], y=ct.bsm[1]/MSY.bsm, pch=0, cex=2, col="red")
        points(x=bk.bsm[nyr], y=ct.bsm[length(ct)]/MSY.bsm[length(MSY.bsm)], pch=2, cex=2,col="red")
      }
      if(substr(id_file,1,3)=="Sim") points(x=true.Bk,y=ct[nyr]/true.MSY,col="green", cex=3, lwd=2)
      #analysis.plot <- recordPlot()
      
      #save analytic chart to JPEG file
      if (save.plots==TRUE) {
        jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_AN.jpg",sep="")
        if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
        dev.copy(jpeg,jpgfile,
          width = 1024,
          height = 768,
          units = "px",
          pointsize = 18,
          quality = 95,
          res=80,
          antialias="cleartype")
        dev.off()
      }
      
      #---------------------------------------------
      # Plot Management-Graphs if desired ----
      #---------------------------------------------
      if(mgraphs==T) {
        # open window for plot of four panels
        if(grepl("win",tolower(Sys.info()['sysname']))) {windows(14,12)}
        par(mfrow=c(2,2))
        # make margins narrower
        par(mar=c(3.1,4.2,2.1,2.1))
        
        #---------------------
        # plot catch with MSY ----
        #---------------------
        max.y <- max(c(1.1*max(ct.jags),ucl.MSY),na.rm=T)
        plot(x=yr,rep(0,nyr),type="n",ylim=c(0,max.y), bty="l", main=paste("Catch",gsub(":","",gsub("/","-",stock))),
          xlab="",ylab="Catch (1000 tonnes/year)",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
        rect(yr[1],lcl.MSY,yr[nyr],ucl.MSY,col="lightgray", border=NA)
        lines(x=c(yr[1],yr[nyr]),y=c(MSY,MSY),lty="dashed", col="black", lwd=2)
        lines(x=yr, y=ct.jags, lwd=2) #
        text("MSY",x=end.yr-1.5, y=MSY+MSY*0.1, cex = .75)
        
        #----------------------------------------
        # Plot of estimated biomass relative to Bmsy
        #----------------------------------------
        # plot empty frame
        plot(yr, rep(0,nyr),type="n", ylim=c(0,max(c(2, max(ucl.B.Bmsy)))), ylab=expression(B/B[MSY]),xlab="", main="Stock size", bty="l",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
        # plot gray area of uncertainty in predicted biomass
        polygon(c(yr,rev(yr)), c(lcl.B.Bmsy,rev(ucl.B.Bmsy)),col="lightgray", border=NA)
        # plot median biomass
        lines(yr,B.Bmsy,lwd=2)
        # plot lines for Bmsy and 0.5 Bmsy
        lines(x=c(yr[1],yr[nyr]),y=c(1,1), lty="dashed", lwd=1.5)
        lines(x=c(yr[1],yr[nyr]),y=c(0.5,0.5), lty="dotted", lwd=1.5)
        
        # -------------------------------------
        ## Plot of exploitation rate
        # -------------------------------------
        # plot empty frame
        plot(yr, rep(0,nyr),type="n", ylim=c(0,max(c(2,ucl.F.Fmsy))),
          ylab=expression(F/F[MSY]),xlab="", main="Exploitation", bty="l",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
        # plot gray area of uncertainty in predicted exploitation
        polygon(c(yr,rev(yr)), c(lcl.F.Fmsy,rev(ucl.F.Fmsy)),col="lightgray", border=NA)
        # plot median exploitation rate
        lines(x=yr,y=F.Fmsy,lwd=2)
        # plot line for u.msy
        lines(x=c(yr[1],yr[nyr]),y=c(1,1), lty="dashed", lwd=1.5)
        
        # -------------------------------------
        ## plot stock-status graph
        # -------------------------------------
        
        if(FullSchaefer==T & force.cmsy==F) {
          x.F_Fmsy = all.FFmsy.bsm[,nyr]
          y.b_bmsy = all.BBmsy.bsm[,nyr]} else { # use CMSY data
            x.F_Fmsy = all.FFmsy.cmsy[,nyr]
            y.b_bmsy = all.BBmsy.cmsy[,nyr]
          }
        
        kernelF <- ci2d(x.F_Fmsy,y.b_bmsy,nbins=201,factor=2.2,ci.levels=c(0.50,0.80,0.75,0.90,0.95),show="none")
        c1 <- c(-1,100)
        c2 <- c(1,1)
        
        max.x1   <- max(c(2, max(kernelF$contours$"0.95"$x,F.Fmsy),na.rm =T))
        max.x    <- ifelse(max.x1 > 5,min(max(5,F.Fmsy*2),8),max.x1)
        max.y    <- max(max(2,quantile(y.b_bmsy,0.96)))
        
        plot(1000,1000,type="b", xlim=c(0,max.x), ylim=c(0,max.y),lty=3,xlab="",ylab=expression(B/B[MSY]), bty="l",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
        mtext(expression(F/F[MSY]),side=1, line=2.3, cex=1,adj=0.55)
        
        # extract interval information from ci2d object
        # and fill areas using the polygon function
        polygon(kernelF$contours$"0.95",lty=2,border=NA,col="cornsilk4")
        polygon(kernelF$contours$"0.8",border=NA,lty=2,col="grey")
        polygon(kernelF$contours$"0.5",border=NA,lty=2,col="cornsilk2")
        
        ## Add points and trajectory lines
        lines(c1,c2,lty=3,lwd=0.7)
        lines(c2,c1,lty=3,lwd=0.7)
        lines(F.Fmsy,B.Bmsy, lty=1,lwd=1.)
        
        # points(F.Fmsy,B.Bmsy,cex=0.8,pch=4)
        points(F.Fmsy[1],B.Bmsy[1],col=1,pch=22,bg="white",cex=1.5)
        points(F.Fmsy[which(yr==int.yr)],B.Bmsy[which(yr==int.yr)],col=1,pch=21,bg="white",cex=1.5)
        points(F.Fmsy[nyr],B.Bmsy[nyr],col=1,pch=24,bg="white",cex=1.5)
        
        ## Add legend
        legend('topright', inset = .03, c(paste(start.yr),paste(int.yr),paste(end.yr),"50% C.I.","80% C.I.","95% C.I."),
          lty=c(1,1,1,-1,-1,-1),pch=c(22,21,24,22,22,22),pt.bg=c(rep("white",3),"cornsilk2","grey","cornsilk4"),
          col=1,lwd=.8,cex=0.85,pt.cex=c(rep(1.1,3),1.5,1.5,1.5),bty="n",y.intersp = 1.1)
        #End of Biplot
        
      } # end of management graphs
      
      #management.plot <- recordPlot()
      
      # save management chart to JPEG file
      if (save.plots==TRUE & mgraphs==TRUE)  {
        jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_MAN.jpg",sep="")
        if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
        dev.copy(jpeg,jpgfile,
          width = 1024,
          height = 768,
          units = "px",
          pointsize = 18,
          quality = 95,
          res=80,
          antialias="cleartype")
        dev.off()
      }
      
      #---------------------------------------------------------
      #><>MSY: rk.diags plot
      #--------------------------------------------------------
      if(rk.diags==T) {
        # open window for plot of four panels
        if(grepl("win",tolower(Sys.info()['sysname']))) {windows(9,9)}
        # make margins narrower
        par(mfrow=c(1,1),mar=c(4.5,4.5,2,0.5))
        plot(x=ri1, y=ki1, xlim = c(0.95*quantile(ri1,0.001),1.2*quantile(ri1,0.999)),
          ylim = c(0.95*quantile(ki1,0.001),1.2*quantile(ki1,0.999)),
          log="xy", xlab="r", ylab="k (1000 tonnes)", main="r-k diagnostic", pch=".", cex=3, bty="l",
          col=rgb(0,0,1,0.5), cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
        points(ppd.r,ppd.k,pch=16,col=rgb(1,0,0,0.5),cex=0.5)
        points(ri.emp,ki1.emp,pch=16,col=rgb(1,0,1,0.5),cex=0.5)
        points(rs,ks,pch=16,col=rgb(0,1,0,0.9),cex=0.5)
        lines(x=c(prior.r[1],prior.r[2],prior.r[2],prior.r[1],prior.r[1]), # plot original prior range
          y=c(prior.k[1],prior.k[1],prior.k[2],prior.k[2],prior.k[1]),
          lty="dotted")
        legend("topright",c("Logistic r-k","Empirical r-k","JAGS r-k","Posterior r-k"),pt.cex = 1.2,pch=15,
          col=c(rgb(0,0,1,0.7),rgb(1,0,1,0.7),rgb(1,0,0,0.7),rgb(0,1,0,1)),bty="n")
        
        if (save.plots==TRUE) {
          jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_rk_Diags.jpg",sep="")
          if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
          dev.copy(jpeg,jpgfile,
            width = 768,
            height = 768,
            units = "px",
            pointsize = 18,
            quality = 95,
            res=80,
            antialias="cleartype")
          dev.off()
        }
      }
      #----------------------------------------------------------
      #><> Optional prior - posterior plots
      #---------------------------------------------------------
      if(pp.plot==T) {
        # open window for plot of four panels
        if(grepl("win",tolower(Sys.info()['sysname']))) {windows(17,12)}
        # make margins narrower
        par(mfrow=c(2,3),mar=c(4.5,4.5,2,0.5))
        greycol = c(grey(0.7,0.5),grey(0.3,0.5)) # changed 0.6 to 0.7
        
        # plot PP-diagnostics for CMSY
        # r
        rk <- exp(mvn(n=5000,mean.log.r=mean.log.r,sd.log.r=sd.log.r,mean.log.k=mean.log.k,sd.log.k=sd.log.k))
        pp.lab = "r"
        rpr = sort(rk[,1])
        post = rs
        prior <-dlnorm(sort(rpr),meanlog = mean.log.r, sdlog = sd.log.r) #><>HW now pdf
        
        # generic ><>HW streamlined GP to check
        nmc = length(post)
        pdf = stats::density(post,adjust=2)
        plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
          yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
        polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
        polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
        PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
        PPVM = round(mean(post)/mean(prior),2)
        pp = c(paste("PPVR =",PPVR))
        legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
        legend("topright",pp,cex=1.4,bty="n")
        
        # k
        pp.lab = "k (1000 tonnes)"
        rpr = sort(rk[,2])
        post = ks
        prior <-dlnorm(sort(rpr),meanlog = mean.log.k, sdlog = sd.log.k) #><>HW now pdf
        # generic ><>HW streamlined GP to check
        nmc = length(post)
        pdf = stats::density(post,adjust=2)
        plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
          yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
        polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
        polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
        PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
        PPVM = round(mean(post)/mean(prior),2)
        pp = c(paste("PPVR =",PPVR))
        legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
        legend("topright",pp,cex=1.4,bty="n")
        
        # Header
        mtext(paste0("CMSY prior & posterior distributions for ",stock),  side=3,cex=1.5)
        
        # MSY
        pp.lab = "MSY (1000 tonnes/year)"
        rpr = sort(rk[,1]*rk[,2]/4)
        post = rs*ks/4
        prior <-dlnorm(sort(rpr),meanlog = mean(log(rpr)), sdlog = sd(log(rpr))) #><>HW now pdf
        prand <- rlnorm(2000,meanlog = mean(log(rpr)), sdlog = sd(log(rpr)))
        # generic ><>HW streamlined GP to check
        nmc = length(post)
        pdf = stats::density(post,adjust=2)
        plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
          yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
        polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
        polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
        PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
        PPVM = round(mean(post)/mean(prand),2)
        pp = c(paste("PPVR =",PPVR))
        legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
        legend("topright",pp,cex=1.4,bty="n")
        
        #><> bk beta priors
        bk.beta = (beta.prior(b.prior))
        
        # bk1
        pp.lab=paste0("B/k ",yr[1])
        post = all.bk.cmsy[,1]
        nmc = length(post)
        rpr = seq(0.5*startbio[1],startbio[2]*1.5,0.005)
        pdf = stats::density(post,adjust=2)
        prand <- sort(rbeta(2000,bk.beta[1,1], bk.beta[2,1]))
        prior <-dbeta(sort(prand),bk.beta[1,1], bk.beta[2,1]) #><>HW now pdf
        #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
        plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
        #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
        polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
        polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
        PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
        PPVM = round(mean(post)/mean(prior),2)
        pp = c(paste("PPVR =",PPVR))
        legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
        legend("topright",pp,cex=1.4,bty="n")
        
        # bk2
        pp.lab=paste0("B/k ", int.yr)
        post = all.bk.cmsy[,which(int.yr==yr)]
        rpr = seq(0.5*intbio[1],intbio[2]*1.5,0.005)
        pdf = stats::density(post,adjust=2)
        prand <- sort(rbeta(2000,bk.beta[1,2], bk.beta[2,2]))
        prior <-dbeta(sort(prand),bk.beta[1,2], bk.beta[2,2]) #><>HW now pdf
        #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
        plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
        #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
        polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
        polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
        PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
        PPVM = round(mean(post)/mean(prior),2)
        pp = c(paste("PPVR =",PPVR))
        legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
        legend("topright",pp,cex=1.4,bty="n")
        
        # bk3
        pp.lab=paste0("B/k ",yr[length(yr)])
        post = all.bk.cmsy[,length(yr)]
        rpr = seq(0.5*endbio[1],endbio[2]*1.5,0.005)
        pdf = stats::density(post,adjust=2)
        prand <- sort(rbeta(2000,bk.beta[1,3], bk.beta[2,3]))
        prior <-dbeta(sort(prand),bk.beta[1,3], bk.beta[2,3]) #><>HW now pdf
        #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
        plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,prand,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
        #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
        polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
        polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
        PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
        PPVM = round(mean(post)/mean(prior),2)
        pp = c(paste("PPVR =",PPVR))
        legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
        legend("topright",pp,cex=1.4,bty="n")
        
        #save analytic chart to JPEG file
        if (save.plots==TRUE) {
          jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_PP_CMSY.jpg",sep="")
          if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
          dev.copy(jpeg,jpgfile,
            width = 1024,
            height = 768,
            units = "px",
            pointsize = 18,
            quality = 95,
            res=80,
            antialias="cleartype")
          dev.off()
        }
        
        # plot PP diagnostics for BSM if available
        if(FullSchaefer==T & force.cmsy==F){ # BSM PLOT
          # open window for plot of four panels
          if(grepl("win",tolower(Sys.info()['sysname']))) {windows(17,12)}
          # make margins narrower
          par(mfrow=c(2,3),mar=c(4.5,4.5,2,0.5))
          greycol = c(grey(0.7,0.5),grey(0.3,0.5))
          
          # r
          rk <- exp(mvn(n=5000,mean.log.r=mean.log.r,sd.log.r=sd.log.r,mean.log.k=mean.log.k,sd.log.k=sd.log.k))
          pp.lab = "r"
          rpr = sort(rk[,1])
          post = rs.bsm
          prior <-dlnorm(sort(rpr),meanlog = mean.log.r, sdlog = sd.log.r) #><>HW now pdf
          
          # generic ><>HW streamlined GP to check
          nmc = length(post)
          pdf = stats::density(post,adjust=2)
          plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
            yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
          polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
          polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
          PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
          PPVM = round(mean(post)/mean(prior),2)
          pp = c(paste("PPVR =",PPVR))
          legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
          legend("topright",pp,cex=1.4,bty="n")
          
          # k
          pp.lab = "k (1000 tonnes)"
          rpr = sort(rk[,2])
          post = ks.bsm
          prior <-dlnorm(sort(rpr),meanlog = mean.log.k, sdlog = sd.log.k) #><>HW now pdf
          # generic ><>HW streamlined GP to check
          nmc = length(post)
          pdf = stats::density(post,adjust=2)
          plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
            yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
          polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
          polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
          PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
          PPVM = round(mean(post)/mean(prior),2)
          pp = c(paste("PPVR =",PPVR))
          legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
          legend("topright",pp,cex=1.4,bty="n")
          
          # Header
          mtext(paste0("BSM prior & posterior distributions for ",stock),  side=3,cex=1.5)
          
          # MSY
          pp.lab = "MSY (1000 tonnes/year)"
          rpr = sort(rk[,1]*rk[,2]/4)
          post = rs.bsm*ks.bsm/4
          prior <-dlnorm(sort(rpr),meanlog = mean(log(rpr)), sdlog = sd(log(rpr))) #><>HW now pdf
          # generic ><>HW streamlined GP to check
          nmc = length(post)
          pdf = stats::density(post,adjust=2)
          plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
            yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
          polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
          polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
          PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
          PPVM = round(mean(post)/mean(prior),2)
          pp = c(paste("PPVR =",PPVR))
          legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
          legend("topright",pp,cex=1.4,bty="n")
          
          # bk1
          pp.lab=paste0("B/k ",yr[1])
          post = all.bk.bsm[,1]
          nmc = length(post)
          rpr = seq(0.5*startbio[1],startbio[2]*1.5,0.005)
          pdf = stats::density(post,adjust=2)
          prand <- sort(rbeta(2000,bk.beta[1,1], bk.beta[2,1]))
          prior <-dbeta(sort(prand),bk.beta[1,1], bk.beta[2,1]) #><>HW now pdf
          #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
          plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
          #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
          polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
          polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
          PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
          PPVM = round(mean(post)/mean(prior),2)
          pp = c(paste("PPVR =",PPVR))
          legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
          legend("topright",pp,cex=1.4,bty="n")
          
          # bk2
          pp.lab=paste0("B/k ", int.yr)
          post = all.bk.bsm[,which(int.yr==yr)]
          rpr = seq(0.5*intbio[1],intbio[2]*1.5,0.005)
          pdf = stats::density(post,adjust=2)
          prand <- sort(rbeta(2000,bk.beta[1,2], bk.beta[2,2]))
          prior <-dbeta(sort(prand),bk.beta[1,2], bk.beta[2,2]) #><>HW now pdf
          #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
          plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
          #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
          if(nbk>1) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
          if(nbk==1) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),lty=2)
          polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
          PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
          PPVM = round(mean(post)/mean(prior),2)
          pp = c(paste("PPVR =",PPVR))
          if(nbk>1){
            legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
            legend("topright",pp,cex=1.4,bty="n")
          } else {
            legend('right',c("Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol[2],bty="n",cex=1.5)
          }
          
          # bk3
          pp.lab=paste0("B/k ",yr[length(yr)])
          post = all.bk.bsm[,length(yr)]
          nmc = length(post)
          rpr = seq(0.5*endbio[1],endbio[2]*1.5,0.005)
          pdf = stats::density(post,adjust=2)
          prand <- sort(rbeta(2000,bk.beta[1,3], bk.beta[2,3]))
          prior <-dbeta(sort(prand),bk.beta[1,3], bk.beta[2,3]) #><>HW now pdf
          #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
          plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,prand,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
          #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
          if(nbk>2) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
          if(nbk<3) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),lty=2)
          polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
          PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
          PPVM = round(mean(post)/mean(prior),2)
          pp = c(paste("PPVR =",PPVR))
          if(nbk>2){
            legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
            legend("topright",pp,cex=1.4,bty="n")
          } else {
            legend('right',c("Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol[2],bty="n",cex=1.5)
          }
          
          #save analytic chart to JPEG file
          if (save.plots==TRUE) {
            jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_PP_BSM.jpg",sep="")
            if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
            dev.copy(jpeg,jpgfile,
              width = 1024,
              height = 768,
              units = "px",
              pointsize = 18,
              quality = 95,
              res=80,
              antialias="cleartype")
            dev.off()
          }
        } # end of BSM plot
      } # End of posterior/prior plot
      
      #----------------------------------------------------------
      #><> Optional BSM diagnostic plot
      #---------------------------------------------------------
      if(BSMfits.plot==T & FullSchaefer==T & force.cmsy==F){
        #---------------------------------------------
        # open window for plot of four panels
        if(grepl("win",tolower(Sys.info()['sysname']))) {windows(9,6)}
        # make margins narrower
        par(mfrow=c(2,2),mar=c(3.1,4.1,2.1,2.1),cex=1)
        cord.x <- c(yr,rev(yr))
        # Observed vs Predicted Catch
        cord.y<-c(lcl.ct.jags,rev(ucl.ct.jags))
        plot(yr,ct,type="n",ylim=c(0,max(ct.jags,na.rm=T)),lty=1,lwd=1.3,xlab="Year",
          ylab=paste0("Catch (1000 tonnes)"),main=paste("Catch fit",stock),bty="l")
        polygon(cord.x,cord.y,col="gray",border=0,lty=1)
        lines(yr,ct.jags,lwd=2,col=1)
        points(yr,(ct),pch=21,bg="white",cex=1.)
        legend("topright",c("Observed","Predicted","95%CIs"),pch=c(21,-1,22),pt.cex = c(1,1,1.5),
          pt.bg=c("white",-1,"grey"),lwd=c(-1,2,-1),col=c(1,1,"grey"),bty="n",y.intersp = 0.9)
        
        # Observed vs Predicted CPUE
        cord.y<-c(lcl.cpue.bsm,rev(ucl.cpue.bsm))
        plot(yr,bt,type="n",ylim=c(0,max(c(pred.cpue,bt),na.rm=T)),lty=1,lwd=1.3,xlab="Year",ylab=paste0("cpue"),
          main="cpue fit",bty="l")
        polygon(cord.x,cord.y,col="gray",border=0,lty=1)
        lines(yr,cpue.bsm,lwd=2,col=1)
        points(yr,(bt),pch=21,bg="white",cex=1.)
        legend("topright",c("Observed","Predicted","95%CIs"),pch=c(21,-1,22),pt.cex = c(1,1,1.5),pt.bg=c("white",-1,"grey"),lwd=c(-1,2,-1),col=c(1,1,"grey"),bty="n",y.intersp = 0.9)
        
        # Process error log-biomass
        cord.y<-c(lcl.pe.bsm,rev(ucl.pe.bsm))
        plot(yr,rep(0,length(yr)),type="n",ylim=c(-max(c(abs(pred.pe),0.2),na.rm=T),max(c(abs(pred.pe),0.2),na.rm=T)),lty=1,lwd=1.3,xlab="Year",ylab=paste0("Deviation log(B)"),main="Process variation",bty="l")
        polygon(cord.x,cord.y,col="gray",border=0,lty=1)
        abline(h=0,lty=2)
        lines(yr,pe.bsm,lwd=2)
        
        
        #-------------------------------------------------
        # Function to do runs.test and 3 x sigma limits
        #------------------------------------------------
        runs.sig3 <- function(x,type="resid") {
          if(type=="resid"){mu = 0}else{mu = mean(x, na.rm = TRUE)}
          # Average moving range
          mr  <- abs(diff(x - mu))
          amr <- mean(mr, na.rm = TRUE)
          # Upper limit for moving ranges
          ulmr <- 3.267 * amr
          # Remove moving ranges greater than ulmr and recalculate amr, Nelson 1982
          mr  <- mr[mr < ulmr]
          amr <- mean(mr, na.rm = TRUE)
          # Calculate standard deviation, Montgomery, 6.33
          stdev <- amr / 1.128
          # Calculate control limits
          lcl <- mu - 3 * stdev
          ucl <- mu + 3 * stdev
          if(nlevels(factor(sign(x)))>1){
            runstest = snpar::runs.test(resid)
            pvalue = round(runstest$p.value,3)} else {
              pvalue = 0.001
            }
          
          return(list(sig3lim=c(lcl,ucl),p.runs= pvalue))
        }
        
        # get residuals
        resid = (log(bt)-log(cpue.bsm))[is.na(bt)==F]
        res.yr = yr[is.na(bt)==F]
        runstest = runs.sig3(resid)
        
        # CPUE Residuals with runs test
        plot(yr,rep(0,length(yr)),type="n",ylim=c(min(-0.25,runstest$sig3lim[1]*1.1),max(0.25,runstest$sig3lim[2]*1.1)),lty=1,lwd=1.3,xlab="Year",ylab=expression(log(cpue[obs])-log(cpue[pred])),main="Residual diagnostics",bty="l")
        abline(h=0,lty=2)
        RMSE = sqrt(mean(resid^2)) # Residual mean sqrt error
        if(RMSE>0.1){lims = runstest$sig3lim} else {lims=c(-1,1)}
        cols = c(rgb(1,0,0,0.5),rgb(0,1,0,0.5))[ifelse(runstest$p.runs<0.05,1,2)]
        if(RMSE>=0.1) rect(min(yr),lims[1],max(yr),lims[2],col=cols,border=cols) # only show runs if RMSE >= 0.1
        for(i in 1:length(resid)){
          lines(c(res.yr[i],res.yr[i]),c(0,resid[i]))
        }
        points(res.yr,resid,pch=21,bg=ifelse(resid < lims[1] | resid > lims[2],2,"white"),cex=1)
        
        # save management chart to JPEG file
        if (save.plots==TRUE & FullSchaefer == T & BSMfits.plot==TRUE) {
          jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_bsmfits.jpg",sep="")
          if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
          dev.copy(jpeg,jpgfile,
            width = 1024,
            height = 768,
            units = "px",
            pointsize = 18,
            quality = 95,
            res=80,
            antialias="cleartype")
          dev.off()
        }
      }
      
      
      #-------------------------------------
      # HW Produce optional kobe plot
      #-------------------------------------
      
      if(kobe.plot==T){
        # open window for plot of four panels
        if(grepl("win",tolower(Sys.info()['sysname']))) {windows(7,7)}
        par(mfrow=c(1,1))
        # make margins narrower
        par(mar=c(5.1,5.1,2.1,2.1))
        
        if(FullSchaefer==T & force.cmsy==F) {
          x.F_Fmsy = all.FFmsy.bsm[,nyr]
          y.b_bmsy = all.BBmsy.bsm[,nyr]} else { # use CMSY data
            x.F_Fmsy = all.FFmsy.cmsy[,nyr]
            y.b_bmsy = all.BBmsy.cmsy[,nyr]
          }
        #><>HW better performance if FFmsy = x for larger values
        kernel.temp <- ci2d(x.F_Fmsy,y.b_bmsy,nbins=201,factor=2.2,ci.levels=c(0.50,0.80,0.75,0.90,0.95),show="none")
        kernelF = kernel.temp
        
        max.x1=max.y1   <- max(c(2, max(kernelF$contours$"0.95"$x,F.Fmsy),na.rm =T))
        max.y    <- ifelse(max.x1 > 5,min(max(5,F.Fmsy*2),8),max.x1)
        max.x    <- max(max(2,quantile(y.b_bmsy,0.96)))
        
        # -------------------------------------
        ## KOBE plot building
        # -------------------------------------
        #Create plot
        plot(1000,1000,type="b", xlim=c(0,max.x), ylim=c(0,max.y),lty=3,xlab="",ylab=expression(F/F[MSY]), bty="l",  cex.main = 2, cex.lab = 1.35, cex.axis = 1.35,xaxs = "i",yaxs="i")
        mtext(expression(B/B[MSY]),side=1, line=3, cex=1.3)
        c1 <- c(-1,100)
        c2 <- c(1,1)
        
        # extract interval information from ci2d object
        # and fill areas using the polygon function
        zb2 = c(0,1)
        zf2  = c(1,100)
        zb1 = c(1,100)
        zf1  = c(0,1)
        polygon(c(zb1,rev(zb1)),c(0,0,1,1),col="green",border=0)
        polygon(c(zb2,rev(zb2)),c(0,0,1,1),col="yellow",border=0)
        polygon(c(1,100,100,1),c(1,1,100,100),col="orange",border=0)
        polygon(c(0,1,1,0),c(1,1,100,100),col="red",border=0)
        
        polygon(kernelF$contours$"0.95"[,2:1],lty=2,border=NA,col="cornsilk4")
        polygon(kernelF$contours$"0.8"[,2:1],border=NA,lty=2,col="grey")
        polygon(kernelF$contours$"0.5"[,2:1],border=NA,lty=2,col="cornsilk2")
        points(B.Bmsy,F.Fmsy,pch=16,cex=1)
        lines(c1,c2,lty=3,lwd=0.7)
        lines(c2,c1,lty=3,lwd=0.7)
        lines(B.Bmsy,F.Fmsy, lty=1,lwd=1.)
        points(B.Bmsy[1],F.Fmsy[1],col=1,pch=22,bg="white",cex=1.5)
        points(B.Bmsy[which(yr==int.yr)],F.Fmsy[which(yr==int.yr)],col=1,pch=21,bg="white",cex=1.5)
        points(B.Bmsy[nyr],F.Fmsy[nyr],col=1,pch=24,bg="white",cex=1.5)
        # Get Propability
        Pr.green = sum(ifelse(y.b_bmsy>1 & x.F_Fmsy<1,1,0))/length(y.b_bmsy)*100
        Pr.red = sum(ifelse(y.b_bmsy<1 & x.F_Fmsy>1,1,0))/length(y.b_bmsy)*100
        Pr.yellow = sum(ifelse(y.b_bmsy<1 & x.F_Fmsy<1,1,0))/length(y.b_bmsy)*100
        Pr.orange = sum(ifelse(y.b_bmsy>1 & x.F_Fmsy>1,1,0))/length(y.b_bmsy)*100
        
        sel.years = c(yr[sel.yr])
        
        legend('topright',
          c(paste(start.yr),paste(int.yr),paste(end.yr),"50% C.I.","80% C.I.","95% C.I.",paste0(round(c(Pr.red,Pr.yellow,Pr.orange,Pr.green),1),"%")),
          lty=c(1,1,1,rep(-1,8)),pch=c(22,21,24,rep(22,8)),pt.bg=c(rep("white",3),"cornsilk2","grey","cornsilk4","red","yellow","orange","green"),
          col=1,lwd=1.1,cex=1.1,pt.cex=c(rep(1.3,3),rep(1.7,3),rep(2.2,4)),bty="n",y.intersp = 1.)
        
        if (save.plots==TRUE & kobe.plot==TRUE) {
          jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_KOBE.jpg",sep="")
          if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
          dev.copy(jpeg,jpgfile,
            width = 1024*0.7,
            height = 1024*0.7,
            units = "px",
            pointsize = 18,
            quality = 95,
            res=80,
            antialias="cleartype")
          dev.off()
        }
      }
      
      #HW Kobe plot end
      #------------------------------------------------------
      # Write cmsy rdata oject (new ><>HW July 2021)
      #------------------------------------------------------
      if(write.rdata==TRUE){
        cmsy = list()
        cmsy$stock = stock
        cmsy$yr = yr
        cmsy$catch = ct
        cmsy$cmsy = list()
        cmsy$cmsy$timeseries = array(data=NA,dim=c(length(yr),3,2),dimnames = list(yr,c("mu","lci","uci"),c("BBmsy","FFmsy")))
        cmsy$cmsy$timeseries[,1,"BBmsy"] =  BBmsy.cmsy
        cmsy$cmsy$timeseries[,2,"BBmsy"] =  lcl.BBmsy.cmsy
        cmsy$cmsy$timeseries[,3,"BBmsy"] =  ucl.BBmsy.cmsy
        cmsy$cmsy$timeseries[,1,"FFmsy"] =  FFmsy.cmsy
        cmsy$cmsy$timeseries[,2,"FFmsy"] =  lcl.FFmsy.cmsy
        cmsy$cmsy$timeseries[,3,"FFmsy"] =  ucl.FFmsy.cmsy
        cmsy$cmsy$brp = t(data.frame(mu=c(r.cmsy,k.cmsy,MSY.cmsy,Bmsy.cmsy,Fmsy.cmsy),lci=c(lcl.r.cmsy,lcl.k.cmsy,lcl.MSY.cmsy,lcl.Bmsy.cmsy,lcl.Fmsy.cmsy),uci=c(ucl.r.cmsy,ucl.k.cmsy,ucl.MSY.cmsy,ucl.Bmsy.cmsy,ucl.Fmsy.cmsy)))
        colnames(cmsy$cmsy$brp) = c("r","k","MSY","Bmsy","Fmsy")
        cmsy$cmsy$rk = data.frame(r=rs,k=ks)
        cmsy$cmsy$kobe = data.frame(BBmsy=all.BBmsy.cmsy[,nyr],FFmsy=all.FFmsy.cmsy[,nyr])
        
        
        if(FullSchaefer==F){
          cmsy$bsm = NULL
        } else {
          cmsy$bsm = list()
          cmsy$bsm$timeseries = array(data=NA,dim=c(length(yr),3,2),dimnames = list(yr,c("mu","lci","uci"),c("BBmsy","FFmsy")))
          cmsy$bsm$timeseries[,1,"BBmsy"] =  BBmsy.bsm
          cmsy$bsm$timeseries[,2,"BBmsy"] =  lcl.BBmsy.bsm
          cmsy$bsm$timeseries[,3,"BBmsy"] =  ucl.BBmsy.bsm
          cmsy$bsm$timeseries[,1,"FFmsy"] =  FFmsy.bsm
          cmsy$bsm$timeseries[,2,"FFmsy"] =  lcl.FFmsy.bsm
          cmsy$bsm$timeseries[,3,"FFmsy"] =  ucl.FFmsy.bsm
          cmsy$bsm$brp = t(data.frame(mu=c(r.bsm,k.bsm,MSY.bsm,Bmsy.bsm,Fmsy.bsm),lci=c(lcl.r.bsm,lcl.k.bsm,lcl.MSY.bsm,lcl.Bmsy.bsm,lcl.Fmsy.bsm),uci=c(ucl.r.bsm,ucl.k.bsm,ucl.MSY.bsm,ucl.Bmsy.bsm,ucl.Fmsy.bsm)))
          colnames(cmsy$bsm$brp) = c("r","k","MSY","Bmsy","Fmsy")
          cmsy$bsm$rk = data.frame(r=rs.bsm,k=ks.bsm)
          cmsy$bsm$kobe = data.frame(BBmsy=all.BBmsy.bsm[,nyr],FFmsy=all.FFmsy.bsm[,nyr])
        } # end of Full Schaefer condition
        
        # save
        save(cmsy,file=paste0("cmsy_",stock,".rdata"))
        
      } #Write Rdata
      
      
      # -------------------------------------
      ## Write results into csv outfile
      # -------------------------------------
      if(write.output == TRUE && retrosp.step==0) { #account for retrospective analysis - write only the last result
        
        # fill catches from 1970 to 2020
        # if leading catches are missing, set them to zero; if trailing catches are missing, set them to NA
        ct.out     <- vector()
        F.Fmsy.out <- vector()
        bt.out     <- vector()
        
        j <- 1
        for(i in 1950 : 2030) {
          if(yr[1]>i) {
            ct.out[j]     <-0
            F.Fmsy.out[j] <-0
            bt.out[j]     <-2*Bmsy
          } else {
            if(i>yr[length(yr)]) {
              ct.out[j]     <-NA
              F.Fmsy.out[j] <-NA
              bt.out[j]     <-NA } else {
                ct.out[j]     <- ct.raw[yr==i]
                F.Fmsy.out[j] <- F.Fmsy[yr==i]
                bt.out[j]     <- B[yr==i]}
          }
          j=j+1
        }
        
        # write data into csv file
        output = data.frame(as.character(cinfo$Group[cinfo$Stock==stock]),
          as.character(cinfo$Region[cinfo$Stock==stock]),
          as.character(cinfo$Subregion[cinfo$Stock==stock]),
          as.character(cinfo$Name[cinfo$Stock==stock]),
          cinfo$ScientificName[cinfo$Stock==stock],
          stock, start.yr, end.yr, start.yr.new, btype,length(bt[is.na(bt)==F]),
          ifelse(FullSchaefer==T,yr[which(bt>0)[1]],NA),
          ifelse(FullSchaefer==T,yr[max(which(bt>0))],NA),
          ifelse(FullSchaefer==T,min(bt[is.na(bt)==F],na.rm=T),NA),
          ifelse(FullSchaefer==T,max(bt[is.na(bt)==F],na.rm=T),NA),
          ifelse(FullSchaefer==T,yr[which.min(bt)],NA),
          ifelse(FullSchaefer==T,yr[which.max(bt)],NA),
          endbio[1],endbio[2],
          ifelse(FullSchaefer==T,q.prior[1],NA),
          ifelse(FullSchaefer==T,q.prior[2],NA),
          max(ct.raw),MSY.pr,mean(ct.raw[(nyr-4):nyr]),sd(ct.raw[(nyr-4):nyr]),ct.raw[nyr],
          min(ct),max(ct),mean(ct),gm.prior.r,
          ifelse(FullSchaefer==T,MSY.bsm,NA), # full Schaefer
          ifelse(FullSchaefer==T,lcl.MSY.bsm,NA),
          ifelse(FullSchaefer==T,ucl.MSY.bsm,NA),
          ifelse(FullSchaefer==T,r.bsm,NA),
          ifelse(FullSchaefer==T,lcl.r.bsm,NA),
          ifelse(FullSchaefer==T,ucl.r.bsm,NA),
          ifelse(FullSchaefer==T,log.r.var,NA),
          ifelse(FullSchaefer==T,k.bsm,NA),
          ifelse(FullSchaefer==T,lcl.k.bsm,NA),
          ifelse(FullSchaefer==T,ucl.k.bsm,NA),
          ifelse(FullSchaefer==T,log.k.var,NA),
          ifelse(FullSchaefer==T,log.kr.cor,NA),
          ifelse(FullSchaefer==T,log.kr.cov,NA),
          ifelse(FullSchaefer==T, q.bsm,NA),
          ifelse(FullSchaefer==T,lcl.q.bsm,NA),
          ifelse(FullSchaefer==T,ucl.q.bsm,NA),
          ifelse(FullSchaefer==T,bk.bsm[nyr],B.Bmsy[nyr]/2), # last B/k JAGS
          ifelse(FullSchaefer==T,lcl.bk.bsm[nyr],NA),
          ifelse(FullSchaefer==T,ucl.bk.bsm[nyr],NA),
          ifelse(FullSchaefer==T,bk.bsm[1],B.Bmsy[1]/2), # first B/k JAGS
          ifelse(FullSchaefer==T,lcl.bk.bsm[1],NA),
          ifelse(FullSchaefer==T,ucl.bk.bsm[1],NA),
          ifelse(FullSchaefer==T,bk.bsm[yr==int.yr],B.Bmsy[yr==int.yr]/2), # int year B/k JAGS
          ifelse(FullSchaefer==T,lcl.bk.bsm[yr==int.yr],NA),
          ifelse(FullSchaefer==T,ucl.bk.bsm[yr==int.yr],NA),
          int.yr, # int year
          ifelse(FullSchaefer==T,FFmsy.bsm[nyr],NA), # last F/Fmsy JAGS
          r.cmsy, lcl.r.cmsy, ucl.r.cmsy, # CMSY r
          k.cmsy, lcl.k.cmsy, ucl.k.cmsy, # CMSY k
          MSY.cmsy, lcl.MSY.cmsy, ucl.MSY.cmsy, # CMSY MSY
          bk.cmsy[nyr],lcl.bk.cmsy[nyr],ucl.bk.cmsy[nyr], # CMSY B/k in last year with catch data
          bk.cmsy[1],lcl.bk.cmsy[1],ucl.bk.cmsy[1], # CMSY B/k in first year
          bk.cmsy[yr==int.yr],lcl.bk.cmsy[yr==int.yr],ucl.bk.cmsy[yr==int.yr], # CMSY B/k in intermediate year
          FFmsy.cmsy[nyr],lcl.FFmsy.cmsy[nyr],ucl.FFmsy.cmsy[nyr],
          Fmsy,lcl.Fmsy,ucl.Fmsy,Fmsy.adj[nyr],lcl.Fmsy.adj[nyr],ucl.Fmsy.adj[nyr],
          MSY,lcl.MSY,ucl.MSY,Bmsy,lcl.Bmsy,ucl.Bmsy,
          B[nyr], lcl.B[nyr], ucl.B[nyr], B.Bmsy[nyr], lcl.B.Bmsy[nyr], ucl.B.Bmsy[nyr],
          Ft[nyr], lcl.Ft[nyr], ucl.Ft[nyr], F.Fmsy[nyr], lcl.F.Fmsy[nyr], ucl.F.Fmsy[nyr],
          ifelse(is.na(sel.yr)==F,B.sel,NA),
          ifelse(is.na(sel.yr)==F,B.Bmsy.sel,NA),
          ifelse(is.na(sel.yr)==F,F.sel,NA),
          ifelse(is.na(sel.yr)==F,F.Fmsy.sel,NA),
          ct.out[1],ct.out[2],ct.out[3],ct.out[4],ct.out[5],ct.out[6],ct.out[7],ct.out[8],ct.out[9],ct.out[10],          # 1950-1959
          ct.out[11],ct.out[12],ct.out[13],ct.out[14],ct.out[15],ct.out[16],ct.out[17],ct.out[18],ct.out[19],ct.out[20], # 1960-1969
          ct.out[21],ct.out[22],ct.out[23],ct.out[24],ct.out[25],ct.out[26],ct.out[27],ct.out[28],ct.out[29],ct.out[30], # 1970-1979
          ct.out[31],ct.out[32],ct.out[33],ct.out[34],ct.out[35],ct.out[36],ct.out[37],ct.out[38],ct.out[39],ct.out[40], # 1980-1989
          ct.out[41],ct.out[42],ct.out[43],ct.out[44],ct.out[45],ct.out[46],ct.out[47],ct.out[48],ct.out[49],ct.out[50], # 1990-1999
          ct.out[51],ct.out[52],ct.out[53],ct.out[54],ct.out[55],ct.out[56],ct.out[57],ct.out[58],ct.out[59],ct.out[60], # 2000-2009
          ct.out[61],ct.out[62],ct.out[63],ct.out[64],ct.out[65],ct.out[66],ct.out[67],ct.out[68],ct.out[69],ct.out[70], # 2010-2019
          ct.out[71],ct.out[72],ct.out[73],ct.out[74],ct.out[75],ct.out[76],ct.out[77],ct.out[78],ct.out[79],ct.out[80],ct.out[81], # 2020-2030
          F.Fmsy.out[1],F.Fmsy.out[2],F.Fmsy.out[3],F.Fmsy.out[4],F.Fmsy.out[5],F.Fmsy.out[6],F.Fmsy.out[7],F.Fmsy.out[8],F.Fmsy.out[9],F.Fmsy.out[10], # 1950-1959
          F.Fmsy.out[11],F.Fmsy.out[12],F.Fmsy.out[13],F.Fmsy.out[14],F.Fmsy.out[15],F.Fmsy.out[16],F.Fmsy.out[17],F.Fmsy.out[18],F.Fmsy.out[19],F.Fmsy.out[20], # 1960-1969
          F.Fmsy.out[21],F.Fmsy.out[22],F.Fmsy.out[23],F.Fmsy.out[24],F.Fmsy.out[25],F.Fmsy.out[26],F.Fmsy.out[27],F.Fmsy.out[28],F.Fmsy.out[29],F.Fmsy.out[30], # 1970-1979
          F.Fmsy.out[31],F.Fmsy.out[32],F.Fmsy.out[33],F.Fmsy.out[34],F.Fmsy.out[35],F.Fmsy.out[36],F.Fmsy.out[37],F.Fmsy.out[38],F.Fmsy.out[39],F.Fmsy.out[40], # 1980-1989
          F.Fmsy.out[41],F.Fmsy.out[42],F.Fmsy.out[43],F.Fmsy.out[44],F.Fmsy.out[45],F.Fmsy.out[46],F.Fmsy.out[47],F.Fmsy.out[48],F.Fmsy.out[49],F.Fmsy.out[50], # 1990-1999
          F.Fmsy.out[51],F.Fmsy.out[52],F.Fmsy.out[53],F.Fmsy.out[54],F.Fmsy.out[55],F.Fmsy.out[56],F.Fmsy.out[57],F.Fmsy.out[58],F.Fmsy.out[59],F.Fmsy.out[60], # 2000-2009
          F.Fmsy.out[61],F.Fmsy.out[62],F.Fmsy.out[63],F.Fmsy.out[64],F.Fmsy.out[65],F.Fmsy.out[66],F.Fmsy.out[67],F.Fmsy.out[68],F.Fmsy.out[69],F.Fmsy.out[70], # 2010-2019
          F.Fmsy.out[71],F.Fmsy.out[72],F.Fmsy.out[73],F.Fmsy.out[74],F.Fmsy.out[75],F.Fmsy.out[76],F.Fmsy.out[77],F.Fmsy.out[78],F.Fmsy.out[79],F.Fmsy.out[80],F.Fmsy.out[81], # 2020-2030
          bt.out[1],bt.out[2],bt.out[3],bt.out[4],bt.out[5],bt.out[6],bt.out[7],bt.out[8],bt.out[9],bt.out[10],           # 1950-1959
          bt.out[11],bt.out[12],bt.out[13],bt.out[14],bt.out[15],bt.out[16],bt.out[17],bt.out[18],bt.out[19],bt.out[20],  # 1960-1969
          bt.out[21],bt.out[22],bt.out[23],bt.out[24],bt.out[25],bt.out[26],bt.out[27],bt.out[28],bt.out[29],bt.out[30],  # 1970-1979
          bt.out[31],bt.out[32],bt.out[33],bt.out[34],bt.out[35],bt.out[36],bt.out[37],bt.out[38],bt.out[39],bt.out[40],  # 1980-1989
          bt.out[41],bt.out[42],bt.out[43],bt.out[44],bt.out[45],bt.out[46],bt.out[47],bt.out[48],bt.out[49],bt.out[50],  # 1990-1999
          bt.out[51],bt.out[52],bt.out[53],bt.out[54],bt.out[55],bt.out[56],bt.out[57],bt.out[58],bt.out[59],bt.out[60],  # 2000-2009
          bt.out[61],bt.out[62],bt.out[63],bt.out[64],bt.out[65],bt.out[66],bt.out[67],bt.out[68],bt.out[69],bt.out[70],  # 2010-2019
          bt.out[71],bt.out[72],bt.out[73],bt.out[74],bt.out[75],bt.out[76],bt.out[77],bt.out[78],bt.out[79],bt.out[80],bt.out[81]) # 2020-2030
        
        write.table(output, file=outfile, append = T, sep = ",",
          dec = ".", row.names = FALSE, col.names = FALSE)
      }
      
      #----------------------------------------------------------------------------------
      # The code below creates a report in PDF format if write.pdf is TRUE ----
      #----------------------------------------------------------------------------------
      ## To generate reports in PDF format, install a LaTeX program. For Windows, you can use https://miktex.org/howto/install-miktex (restart after installation)
      ## Set write.pdf to 'TRUE' if you want pdf output.
      
      options(tinytex.verbose = TRUE)
      
      # Using MarkdownReports, this creates a markdown file for each stock then using rmarkdown to render each markdown file into a pdf file.
      if(write.pdf == TRUE) {
        library(knitr)
        library(tinytex)
        
        docTemplate <- "\\documentclass[12pt,a4paper]{article}
    \\setlength\\parindent{0pt}
    \\usepackage{geometry}
    \\usepackage{graphicx}
    \\usepackage{grffile}
    \\geometry{margin=0.5in}
    \\begin{document}

    \\section*{#TITLE#}


    #INTRO#

    \\begin{figure}[ht]
    \\centering
    \\includegraphics[width=1.00\\textwidth ext=.jpg type=jpg]{#IMAGE1#}
    \\end{figure}

    #MANAGEMENT#

    \\pagebreak

    \\begin{figure}[ht]
    \\centering
    \\includegraphics[width=1.00\\textwidth ext=.jpg type=jpg]{#IMAGE2#}
    \\end{figure}

    #ANALYSIS#

    \\end{document}"
        
        title = gsub(":","",gsub("/","-",cinfo$Name[cinfo$Stock==stock]))
        
        intro = (paste("Species: \\\\emph{",cinfo$ScientificName[cinfo$Stock==stock],"}, Stock code: ",
          gsub(":","",gsub("/","-",stock)), sep=""))
        intro = (paste(intro,"\n\n","Region: ",gsub(":","",gsub("/","-",cinfo$Region[cinfo$Stock==stock])), sep=""))
        intro = (paste(intro,"\n\n","Marine Ecoregion: ",gsub(":","",gsub("/","-",cinfo$Subregion[cinfo$Stock==stock])), sep="" ))
        intro = (paste(intro,"\n\n","Reconstructed catch data used from years ", min(yr)," - ", max(yr),sep=""))
        intro = (paste(intro,"\n\n","For figure captions and method see http://www.seaaroundus.org/cmsy-method"))
        
        
        docTemplate<-gsub("#TITLE#", title, docTemplate)
        docTemplate<-gsub("#INTRO#", intro, docTemplate)
        
        
        management_text<-paste("\\\\textbf{Results for management (based on",ifelse(FullSchaefer==F | force.cmsy==T,"CMSY","BSM"),"analysis)}\\\\\\\\")
        management_text<-(paste(management_text,"\n\n","Fmsy = ",format(Fmsy, digits =3),", 95% CL = ",format(lcl.Fmsy, digits =3)," - ",format(ucl.Fmsy, digits =3)," (if B $>$ 1/2 Bmsy then Fmsy = 0.5 r)", sep=""))
        management_text<-(paste(management_text,"\n\n","Fmsy = ",format(Fmsy.adj[nyr], digits =3),", 95% CL = ",format(lcl.Fmsy.adj[nyr], digits =3)," - ",format(ucl.Fmsy.adj[nyr], digits =3)," (r and Fmsy are linearly reduced if B $<$ 1/2 Bmsy)",sep=""))
        management_text<-(paste(management_text,"\n\n","MSY = ",format(MSY, digits =3),",  95% CL = ",format(lcl.MSY, digits =3)," - ",format(ucl.MSY, digits =3),'; Bmsy = ',format(Bmsy, digits =3),",  95% CL = ",format(lcl.Bmsy, digits =3)," - ",format(ucl.Bmsy, digits =3)," (1000 tonnes)",sep=""))
        management_text<-(paste(management_text,"\n\n","Biomass in last year = ",format(B[nyr], digits =3),", 95% CL = ", format(lcl.B[nyr], digits =3), " - ",format(ucl.B[nyr], digits =3)," (1000 tonnes)",sep=""))
        management_text<-(paste(management_text,"\n\n","B/Bmsy in last year = " ,format(B.Bmsy[nyr], digits =3),", 95% CL = ", format(lcl.B.Bmsy[nyr], digits =3), " - ",format(ucl.B.Bmsy[nyr], digits =3),sep=""))
        management_text<-(paste(management_text,"\n\n","Fishing mortality in last year = ",format(Ft[nyr], digits =3),", 95% CL =", format(lcl.Ft[nyr], digits =3), " - ",format(ucl.Ft[nyr], digits =3),sep=""))
        management_text<-(paste(management_text,"\n\n","F/Fmsy  = ",format(F.Fmsy[nyr], digits =3),", 95% CL = ", format(lcl.F.Fmsy[nyr], digits =3), " - ",format(ucl.F.Fmsy[nyr], digits =3),sep=""))
        management_text<-(paste(management_text,"\n\n","Comment:", gsub(":","",gsub("/","",comment)), ""))
        docTemplate<-gsub("#MANAGEMENT#", management_text, docTemplate)
        
        analysis_text<-(paste("\\\\textbf{Results of CMSY analysis conducted in JAGS}\\\\\\\\",sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","r = ", format(r.cmsy, digits =3),", 95% CL = ", format(lcl.r.cmsy, digits =3), " - ", format(ucl.r.cmsy, digits =3),"; k = ", format(k.cmsy, digits =3),", 95% CL = ", format(lcl.k.cmsy, digits =3), " - ", format(ucl.k.cmsy, digits =3)," (1000 tonnes)",sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","MSY = ", format(MSY.cmsy, digits =3),", 95% CL = ", format(lcl.MSY.cmsy, digits =3), " - ", format(ucl.MSY.cmsy, digits =3)," (1000 tonnes/year)",sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","Relative biomass last year = ", format(bk.cmsy[nyr], digits =3), " k, 95% CL = ", format(lcl.bk.cmsy[nyr], digits =3), " - ", format(ucl.bk.cmsy[nyr], digits =3),sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","Exploitation F/(r/2) in last year = ", format((FFmsy.cmsy)[length(bk.cmsy)-1], digits =3),sep=""))
        
        if(FullSchaefer==T) {
          analysis_text <- paste(analysis_text,"\\\\\\\\")
          analysis_text<-(paste(analysis_text,"\n\n", "\\\\textbf{Results from Bayesian Schaefer model using catch and ",btype,"}\\\\\\\\",sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","r = ", format(r.bsm, digits =3),", 95% CL = ", format(lcl.r.bsm, digits =3), " - ", format(ucl.r.bsm, digits =3),"; k = ", format(k.bsm, digits =3),", 95% CL = ", format(lcl.k.bsm, digits =3), " - ", format(ucl.k.bsm, digits =3),sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","r-k log correlation = ", format(log.kr.cor, digits =3),sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","MSY = ", format(MSY.bsm, digits =3),", 95% CL = ", format(lcl.MSY.bsm, digits =3), " - ", format(ucl.MSY.bsm, digits =3)," (1000 tonnes/year)",sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","Relative biomass in last year = ", format(bk.cmsy[nyr], digits =3), " k, 95% CL = ",format(lcl.bk.cmsy[nyr], digits =3)," - ", format(ucl.bk.cmsy[nyr], digits =3),sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","Exploitation F/(r/2) in last year = ", format((ct.raw[nyr]/(bk.cmsy[nyr]*k.bsm))/(r.bsm/2), digits =3),sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","q = ", format(q.bsm, digits =3),", 95% CL = ", format(lcl.q.bsm, digits =3), " - ", format(ucl.q.bsm, digits =3),sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","Prior range of q = ",format(q.prior[1], digits =3)," - ",format(q.prior[2], digits =3),sep=""))
        }
        # show stock status and exploitation for optional selected year
        if(is.na(sel.yr)==F) {
          analysis_text<-(paste(analysis_text,"\n\n","Stock status and exploitation in ",sel.yr,sep=""))
          analysis_text<-(paste(analysis_text,"\n\n","Biomass = ",format(B.sel, digits =3), ", B/Bmsy = ",format(B.Bmsy.sel, digits =3),", fishing mortality F = ",format(F.sel, digits =3),", F/Fmsy = ",format(F.Fmsy.sel, digits =3),sep=""))
        }
        
        if(btype !="None" & length(bt[is.na(bt)==F])<nab) {
          analysis_text<-(paste(analysis_text,"\n\n","Less than ",nab," years with abundance data available, shown on second axis",sep="")) }
        
        
        analysis_text<-(paste(analysis_text,"\n\n","Relative abundance data type = ", format(btype, digits =3),sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","Prior initial relative biomass = ", format(startbio[1], digits =3) , " - ", format(startbio[2], digits =3),ifelse(is.na(stb.low)==T," default"," expert"),sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","Prior intermediate relative biomass = ", format(intbio[1], digits =3), " - ", format(intbio[2], digits =3), " in year ", int.yr,ifelse(is.na(intb.low)==T," default"," expert"),sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","Prior final relative biomass = ", format(endbio[1], digits =3), " - ", format(endbio[2], digits =3),ifelse(is.na(endb.low)==T,", default"," expert"),sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","Prior range for r = ", format(prior.r[1],digits=2), " - ", format(prior.r[2],digits=2),ifelse(is.na(r.low)==T," default"," expert"),", prior range for k = " , format(prior.k[1], digits =3), " - ", format(prior.k[2], digits =3)," (1000 tonnes) default",sep=""))
        analysis_text<-(paste(analysis_text,"\n\n","Source for relative biomass: \n\n",source,"",sep=""))
        
        docTemplate<-gsub("#ANALYSIS#", analysis_text, docTemplate)
        
        docTemplate<-gsub("_", "\\\\_", docTemplate)
        docTemplate<-gsub("%", "\\\\%", docTemplate)
        
        
        analysischartfile<-paste(gsub(":","",gsub("/","-",stock)),"_AN.jpg",sep="")
        managementchartfile<-paste(gsub(":","",gsub("/","-",stock)),"_MAN.jpg",sep="")
        docTemplate<-gsub("#IMAGE1#", managementchartfile, docTemplate)
        docTemplate<-gsub("#IMAGE2#", analysischartfile, docTemplate)
        
        # unique filenames to prevent error if files exists from previous run
        documentfile<-paste(gsub(":","",gsub("/","-",stock)),substr(as.character(Sys.time()),1,10),"-",sub(":","",substr(as.character(Sys.time()),12,16)),".RnW",sep="") # concatenated hours and minutes added to file name
        cat(docTemplate,file=documentfile,append=F)
        
        knit(documentfile)
        knitr::knit2pdf(documentfile)
        
        cat("PDF document is ",gsub(".RnW",".pdf",documentfile))
        
      }
      # end of loop to write text to file
      
      
      if(close.plots==T) graphics.off() # close on-screen graphics windows after files are saved
      
      FFmsy.retrospective[[retrosp.step+1]]<-F.Fmsy #retrospective analysis
      BBmsy.retrospective[[retrosp.step+1]]<-B.Bmsy #retrospective analysis
      years.retrospective[[retrosp.step+1]]<-yr #retrospective analysis
      
    } #retrospective analysis - end loop
    
    #retrospective analysis plots
    if (retros == T){
      
      if(grepl("win",tolower(Sys.info()['sysname']))) {windows(14,7)}
      par(mfrow=c(1,2), mar=c(4,5,4,5),  oma=c(2,2,2,2))
      
      allyears<-years.retrospective[[1]]
      nyrtotal<-length(allyears)
      legendyears<-c("All years")
      #CHECK IF ALL YEARS HAVE BEEN COMPUTED
      for (ll in 1:4){
        if (ll>length(FFmsy.retrospective)){
          FFmsy.retrospective[[ll]]<-c(0)
          BBmsy.retrospective[[ll]]<-c(0)
        }
        else {
          if(ll>1)
            legendyears<-c(legendyears,allyears[nyrtotal-ll+1])
        }
      }
      
      #PLOT FFMSY RETROSPECTIVE ANALYSIS
      plot(x=allyears[1:nyrtotal],y=FFmsy.retrospective[[1]], main="",
        ylim=c(0,max(max(FFmsy.retrospective[[1]],na.rm=T),
          max(FFmsy.retrospective[[2]],na.rm=T),
          max(FFmsy.retrospective[[3]],na.rm=T),
          max(FFmsy.retrospective[[4]],na.rm=T))),
        lwd=2, xlab="Year", ylab="F/Fmsy", type="l", bty="l",
        cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.5) #, xaxs="i",yaxs="i",xaxt="n",yaxt="n")
      #PLOT ONLY THE TIME SERIES THAT ARE COMPLETE
      if (length(FFmsy.retrospective[[2]])>1 || FFmsy.retrospective[[2]]!=0)
        lines(x=allyears[1:(nyrtotal-1)],y=FFmsy.retrospective[[2]], type = "o", pch=15, col="red")
      if (length(FFmsy.retrospective[[3]])>1 || FFmsy.retrospective[[3]]!=0)
        lines(x=allyears[1:(nyrtotal-2)],y=FFmsy.retrospective[[3]], type = "o", pch=16, col="green")
      if (length(FFmsy.retrospective[[4]])>1 || FFmsy.retrospective[[4]]!=0)
        lines(x=allyears[1:(nyrtotal-3)],y=FFmsy.retrospective[[4]], type = "o", pch=17, col="blue")
      legend("bottomleft", legend = legendyears,
        col=c("black","red", "green", "blue"), lty=1, pch=c(-1,15,16,17))
      #PLOT BBMSY RETROSPECTIVE ANALYSIS
      plot(x=allyears[1:(nyrtotal)],y=BBmsy.retrospective[[1]],main="", ylim=c(0,max(max(BBmsy.retrospective[[1]],na.rm=T),
        max(BBmsy.retrospective[[2]],na.rm=T),
        max(BBmsy.retrospective[[3]],na.rm=T),
        max(BBmsy.retrospective[[4]],na.rm=T))),
        lwd=2, xlab="Year", ylab="B/Bmsy", type="l", bty="l",cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.5) #, xaxs="i",yaxs="i",xaxt="n",yaxt="n")
      if (length(BBmsy.retrospective[[2]])>1 || BBmsy.retrospective[[2]]!=0)
        lines(x=allyears[1:(nyrtotal-1)],y=BBmsy.retrospective[[2]], type = "o", pch=15, col="red")
      if (length(BBmsy.retrospective[[3]])>1 || BBmsy.retrospective[[3]]!=0)
        lines(x=allyears[1:(nyrtotal-2)],y=BBmsy.retrospective[[3]], type = "o", pch=16, col="green")
      if (length(BBmsy.retrospective[[4]])>1 || BBmsy.retrospective[[4]]!=0)
        lines(x=allyears[1:(nyrtotal-3)],y=BBmsy.retrospective[[4]], type = "o", pch=17, col="blue")
      legend("bottomleft", legend = legendyears,
        col=c("black","red", "green", "blue"), lty=1, pch=c(-1,15,16,17))
      
      mtext(paste0("Retrospective analysis for ",stock),  outer = T , cex=1.5)
      
      #save analytic chart to JPEG file
      if (save.plots==TRUE) {
        jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_RetrospectiveAnalysis.jpg",sep="")
        dev.copy(jpeg,jpgfile,
          width = 1024,
          height = 576,
          units = "px",
          pointsize = 10,
          quality = 95,
          res=80,
          antialias="default")
        dev.off()
      }
      
      if(close.plots==T) graphics.off() # close on-screen graphics windows after files are saved
    } #retrospective analysis plots - end
    
  } # end of stocks loop of CMSY++
  #--------------------------------------------------------------------------
  
  
  
  
  #X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X
  #Continuation of stocks loop 
  #X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X
  
  #bind results in out_list
  #breaking down region and category from stock name 
  
  # Extract scenario type (Base or Projection)
  scenario   = str_extract(stk, "Base|Projection")
  # Extract depletion estimation method (Neural Network or BRT)
  bk_method  = str_extract(stk, "NN|BRT")
  # Extract intrinsic growth rate estimation method
  r_method   = str_extract(stk, "Euler|Myers|Smith rebound|Demographic inv")
  # Extract the data source (Freire or Silva)
  source   = str_extract(stk, "Freire|Silva")
  # Extract region (North, Northeast, Southeast, South)
  region     = str_extract(stk, "(?<=_)(NE|SE|N|S)(?=_)")
  # Extract common species name
  name       = str_extract(stk, "brown|pink|seabob|white")
  
  
  biodat<-data.frame(
    stock=stk,name=name,yr=yr,region=region,scenario=scenario,source=source,bk_method=bk_method,r_method=r_method,
    nyr=nyr,start.yr=start.yr,int.yr=int.yr,end.yr=end.yr,
    ucl.B.Bmsy=ucl.B.Bmsy,lcl.B.Bmsy=lcl.B.Bmsy,B.Bmsy=B.Bmsy,
    ucl.F.Fmsy=ucl.F.Fmsy,lcl.F.Fmsy=lcl.F.Fmsy,F.Fmsy=F.Fmsy)
  bio_out= rbind(bio_out,biodat)
  #---------------------------
  
  
  #----------------------------------------------------------------
  # PRIOR–POSTERIOR DIAGNOSTICS (r, k, B/k, MSY, and final metrics)
  #----------------------------------------------------------------
  
  #-----------------------------------------------
  # Priors for r and k
  #-----------------------------------------------
  rk <- exp(mvn(n = length(rs),
                mean.log.r = mean.log.r,
                sd.log.r = sd.log.r,
                mean.log.k = mean.log.k,
                sd.log.k = sd.log.k))
  
  priorr <- rk[, 1]
  priork <- rk[, 2]
  postr  <- rs
  postk  <- ks
  
  rk_dat<- data.frame(
    stock=stk,name=name,region=region,scenario=scenario,source=source,bk_method=bk_method,r_method=r_method,
    priorr=priorr,priork=priork,postr=postr,postk=postk)
  rk_out= rbind(rk_out,rk_dat)
  
  #-----------------------------------------------
  # Diagnostics for r
  #-----------------------------------------------
  prior_log_r <- log(priorr)
  post_log_r  <- log(postr)
  
  PPVRr <- round(var(post_log_r, na.rm = TRUE) / var(prior_log_r, na.rm = TRUE), 3)
  RUr   <- round(1 - sd(post_log_r, na.rm = TRUE) / sd(prior_log_r, na.rm = TRUE), 3)
  
  dens_prior_r <- density(prior_log_r, n = 512)
  dens_post_r  <- density(post_log_r, n = 512)
  xs <- seq(max(min(dens_prior_r$x), min(dens_post_r$x)),
            min(max(dens_prior_r$x), max(dens_post_r$x)), length.out = 512)
  p1 <- approx(dens_prior_r$x, dens_prior_r$y, xout = xs)$y
  p2 <- approx(dens_post_r$x, dens_post_r$y, xout = xs)$y
  
  OVLr <- round(sum(pmin(p1, p2)) * mean(diff(xs)), 3)
  BCr  <- round(sum(sqrt(p1 * p2)) * mean(diff(xs)), 3)
  
  # ----------------------------------------------------------------------
  # Prior-predictive checks trajectories using CMSY/BSM Schaefer dynamics
  # ----------------------------------------------------------------------
  simulate_prior_cmsy <- function(prior_r, prior_k,
                                  C_t,                     # vector of catches (same length = nyrs)
                                  startbio = c(0.8, 1.0),  # vector c(min,max) used to draw alpha ~ lognormal like in BSM
                                  viable_range = c(0.2, 0.7), # B/k viable interval (depletion range)
                                  eps = 0.01,
                                  proc_sd = 0.1,           # sd on log scale for process error
                                  n_sims = 6000) {
    
    nyrs <- length(C_t)
    
    # If prior vectors shorter/longer than n_sims, sample with replacement
    if(length(prior_r) < n_sims) prior_r <- sample(prior_r, n_sims, replace = TRUE)
    if(length(prior_k) < n_sims) prior_k <- sample(prior_k, n_sims, replace = TRUE)
    if(length(prior_r) > n_sims) prior_r <- sample(prior_r, n_sims, replace = FALSE)
    if(length(prior_k) > n_sims) prior_k <- sample(prior_k, n_sims, replace = FALSE)
    
    # draw initial relative biomass P1 consistent with startbio prior (lognormal-ish as in BSM)
    log.alpha <- log((startbio[1] + startbio[2]) / 2)
    sd.log.alpha <- (log.alpha - log(startbio[1])) / 4
    alpha_samps <- rlnorm(n_sims, meanlog = log.alpha, sdlog = sd.log.alpha)
    
    # storage
    is_viable <- logical(n_sims)
    any_negative_or_na <- logical(n_sims)
    
    for (i in seq_len(n_sims)) {
      r_i <- prior_r[i]
      k_i <- prior_k[i]
      P <- numeric(nyrs)
      P[1] <- max(alpha_samps[i], eps)
      bad_flag <- FALSE
      
      for (t in 2:nyrs) {
        # deterministic next relative biomass according to CMSY/BSM form:
        if (P[t-1] > 0.25) {
          det_next <- P[t-1] + r_i * P[t-1] * (1 - P[t-1]) - (C_t[t-1] / k_i)
        } else {
          # reduced productivity when biomass low
          det_next <- P[t-1] + 4 * P[t-1] * r_i * P[t-1] * (1 - P[t-1]) - (C_t[t-1] / k_i)
        }
        
        det_next <- max(det_next, eps)
        logdet <- log(det_next)
        logP_t <- rnorm(1, mean = logdet, sd = proc_sd)
        P[t] <- exp(logP_t)
        
        if (is.na(P[t]) || !is.finite(P[t])) { bad_flag <- TRUE; break }
        if (P[t] > 10) P[t] <- 10
      } # t
      
      any_negative_or_na[i] <- bad_flag || any(is.na(P))
      
      # check if final B/k is within viable range
      P_final <- P[nyrs]
      is_viable[i] <- P_final >= viable_range[1] & P_final <= viable_range[2]
    } # i
    
    # summarize metrics
    viability_rate <- mean(is_viable, na.rm = TRUE)
    
    result <- list(
      is_viable = is_viable,
      viability_rate = round(viability_rate, 3),
      n_sims = n_sims,
      any_bad = any_negative_or_na
    )
    return(result)
  }
  
  #Simulating trajectories based on prior distributions 
  ppv <- simulate_prior_cmsy(
    prior_r = priorr,
    prior_k = priork,
    C_t = ct,
    startbio = startbio,
    viable_range = endbio,
    proc_sd = 0.15,
    n_sims = 6000
  )
  viab_rate= ppv$viability_rate
  
  #-----------------------------------------------
  # Diagnostics for depletion (B/k)
  #-----------------------------------------------
  postbk  <- all.bk.cmsy[, length(yr)]
  priorbk <- rbeta(length(postbk), bk.beta[1, 3], bk.beta[2, 3])
  
  eps <- 1e-9
  prior_logit_bk <- log(pmin(pmax(priorbk, eps), 1 - eps) / (1 - pmin(pmax(priorbk, eps), 1 - eps)))
  post_logit_bk  <- log(pmin(pmax(postbk, eps), 1 - eps) / (1 - pmin(pmax(postbk, eps), 1 - eps)))
  
  PPVRbk <- round(var(post_logit_bk, na.rm = TRUE) / var(prior_logit_bk, na.rm = TRUE), 3)
  RUbk   <- round(1 - sd(post_logit_bk, na.rm = TRUE) / sd(prior_logit_bk, na.rm = TRUE), 3)
  
  dens_prior_bk <- density(prior_logit_bk, n = 512)
  dens_post_bk  <- density(post_logit_bk, n = 512)
  xs <- seq(max(min(dens_prior_bk$x), min(dens_post_bk$x)),
            min(max(dens_prior_bk$x), max(dens_post_bk$x)), length.out = 512)
  p1 <- approx(dens_prior_bk$x, dens_prior_bk$y, xout = xs)$y
  p2 <- approx(dens_post_bk$x, dens_post_bk$y, xout = xs)$y
  
  OVLbk <- round(sum(pmin(p1, p2)) * mean(diff(xs)), 3)
  BCbk  <- round(sum(sqrt(p1 * p2)) * mean(diff(xs)), 3)
  
  #Another CMSY++ results-- Final Metrics
  #intervals for prior and posteriors for r, k, bk and msy
  priorrlw=round(prior.r[1],2)
  priorrup=round(prior.r[2],2)
  postrlw= round(lcl.r.cmsy,2)
  postrup= round(ucl.r.cmsy,2)
  priorklw=round(prior.k[1])       
  priorkup=round(prior.k[2])     
  postklw=round(lcl.k.cmsy,2)
  postkup=round(ucl.k.cmsy,2)
  priorbklw=round(endbio[1],2)
  priorbkup=round(endbio[2],2)
  postbklw=round(lcl.bk.cmsy[nyr],2)
  postbkup=round(ucl.bk.cmsy[nyr],2)
  priormsylw=round(0.8*MSY.pr,2)
  priormsyup=round(1.2*MSY.pr,2)
  postmsylw=as.numeric(round(lcl.MSY.cmsy,2))
  postmsyup=as.numeric(round(ucl.MSY.cmsy,2))
  postmsyavg= round(MSY.cmsy,2)
  fmsylw= as.numeric(round(lcl.Fmsy.cmsy,2))
  fmsyup= as.numeric(round(ucl.Fmsy.cmsy,2))
  fmsyavg= as.numeric(round(Fmsy.cmsy,2))
  bmsylw= as.numeric(round(lcl.Bmsy.cmsy,2))
  bmsyup= as.numeric(round(ucl.Bmsy.cmsy,2))
  bmsyavg= as.numeric(round(Bmsy.cmsy,2))
  ctlast=round(ct.raw[nyr],2)
  ctavg5=round(mean(ct.raw[(nyr-4):(nyr)]),2)
  ffmsylw=round(lcl.FFmsy.cmsy[nyr],2)
  ffmsyup=round(ucl.FFmsy.cmsy[nyr],2)
  ffmsyavg=round(FFmsy.cmsy[nyr],2)
  bbmsylw=round(lcl.BBmsy.cmsy[nyr],2)
  bbmsyup=round(ucl.BBmsy.cmsy[nyr],2)
  bbmsyavg=round(BBmsy.cmsy[nyr],2)
  
  
  #-----------------------------------------------
  # Combine everything in a SINGLE data frame
  #-----------------------------------------------
  cmsy_dat <- data.frame(
    stock = stk, name = name, region = region, scenario = scenario,
    source = source, bk_method = bk_method, r_method = r_method,
    priorrlw= priorrlw,priorr = mean(priorr), priorrup=priorrup,
    postrlw=postrlw, postr = mean(postr), postrup=postrup,
    priorklw=priorklw, priork = mean(priork), priorkup=priorkup, 
    postklw=postklw,postk = mean(postk),postkup= postkup,
    priorbklw=priorbklw,priorbk = mean(priorbk),priorbkup=priorbkup, 
    postbklw=postbklw,postbk = mean(postbk), postbkup= postbkup,
    priormsylw=priormsylw, priormsy= mean(MSY.pr), priormsyup=priormsyup,
    postmsylw=postmsylw, postmsy=postmsyavg, postmsyup=postmsyup,
    fmsylw=fmsylw, fmsy= fmsyavg, fmsyup= fmsyup,
    bmsylw=bmsylw, bmsy= bmsyavg, bmsyup=bmsyup, 
    ctlast=ctlast,ctavg5=ctavg5,
    ffmsylw=ffmsylw,ffmsy=ffmsyavg,ffmsyup=ffmsyup,
    bbmsylw=bbmsylw,bbmsy=bbmsyavg, bbmsyup=bbmsyup,
    PPVRr = PPVRr, RUr = RUr, OVLr = OVLr, BCr = BCr,
    PPVRbk = PPVRbk, RUbk = RUbk, OVLbk = OVLbk, BCbk = BCbk,
    viab_rate = viab_rate,
    mean.log.r = mean.log.r, sd.log.r = sd.log.r,
    mean.log.k = mean.log.k, sd.log.k = sd.log.k
  )
  
  cmsy_out <- rbind(cmsy_dat, cmsy_out)
  
  #get data to construct Kobe plots compositions (F.msy and B.msy final points)
  kobedat=data.frame(stock=stk,name=name,region=region,scenario=scenario,
                     source = source, bk_method = bk_method, r_method = r_method,
                     x.F_Fmsy=x.F_Fmsy, y.b_bmsy=y.b_bmsy)
  kobe_out= rbind(kobe_out, kobedat)
  
}#end of stocks loop (Silva MLS)


#writing some results..

#rewriting the updated cinfo
write.table(cinfo, file="cinfo_shrimp_updated.csv",
            dec=".",sep=",",row.names = FALSE)

#write kobe out with f/fmsy and b/bmsy series
write.table(bio_out, file = "bio_out.csv", 
            dec=".",sep = ",", row.names = FALSE) 

#write r-k samples priors/posteriors
write.table(rk_out, file = "rk_out.csv", 
            dec=".",sep = ",", row.names = FALSE) 

#write the prior posterior data intervals
write.table(cmsy_out, file = "cmsy_out.csv", 
            dec=".",sep = ",", row.names = FALSE) 

#write kobe out with f/fmsy and b/bmsy series
write.table(kobe_out, file = "kobe_out.csv", 
            dec=".",sep = ",", row.names = FALSE) 
#---------------------------------------------------------------


#-------------------------------------------------------------------
# Projection Scenario

# best models with the best score
# posterior as input in projections
#-------------------------------------------------------------------

# reading the output data
bio_out<- read.csv("bio_out.csv",dec=".",sep=",")
rk_out<- read.csv("rk_out.csv",dec=".",sep=",")
cmsy_out<-read.csv("cmsy_out.csv",dec=".",sep=",")
kobe_out<-read.csv("kobe_out.csv",dec=".",sep=",")
ct_out<- read.csv("cdat_shrimp.csv",dec=".",sep=",")

                #X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X
                #-----------Plot section-----------#
                #X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X


#picking the best tested models
#PRIOR- POSTERIOR PLOTS
cmsy_long <- cmsy_out %>%
  filter(!scenario== "Projection") %>%
  pivot_longer(
    cols = c(RUr, OVLr, viab_rate),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(metric,
                    "RUr" = "RU",
                    "OVLr" = "OVL",
                    "viab_rate" = "VR"
    )
  )

p5 <- cmsy_long %>%
  mutate(method_combo = interaction(bk_method, r_method, sep = " - ")) %>%
  ggplot(aes(x = method_combo, y = value, fill = metric)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, width = 0.6) +
  labs(
    x = "",
    y = "Metric value",
    fill = ""
  ) +
  # scale_fill_manual(values = c(
  #   "RU" = "#56B4E9",
  #   "OVL" = "#E69F00",
  #   "VR" = "#009E73"
  # )) +
  scale_fill_viridis_d()+
  theme_classic(base_size = 15) %+replace%
  theme(
    strip.background = element_blank(), 
    plot.margin = unit(c(0.05, 0.05, 0.05, 11), "mm"), # Tight plot margins
    axis.text.y =element_text(size = 16),
    axis.text.x =element_text(size = 16, angle = 45, vjust =1, hjust = 1),
    strip.text.x = element_text(
      margin = margin(b = 1),  # Bottom margin for facet labels
      size = 16
    ),
    legend.box.margin = margin(t = -40), # Move legend closer
    legend.text = element_text(size=16),
    legend.spacing.y = unit(0.1, "cm"),  # Reduce spacing between legend rows
    legend.position = "bottom"           # Place legend at bottom
  )

p5


#---------------------------------------------------------
# Calculate the best scenarios by group (Silva and Freire)
#---------------------------------------------------------
# Filter only the test scenarios ("Base")
cmsy_base <- cmsy_out %>%
  filter(grepl("_Base", stock))

#Scaling metrics within each 'source' group
cmsy_scaled <- cmsy_base %>%
  group_by(name,region,source) %>%
  mutate(
    RU_sc  = rescale(RUr, to = c(0, 1)),
    OVL_sc = rescale(OVLr, to = c(0, 1)),
    VR_sc  = rescale(viab_rate, to = c(0, 1))
  ) %>%
  ungroup()

# Calculate the weighted final score
cmsy_scored <- cmsy_scaled %>%
  mutate(final_score = 0.2 * RU_sc + 0.2 * OVL_sc + 0.6 * VR_sc)

# Select the best scenario by stock (name + region + source)
best_by_group <- cmsy_scored %>%
  group_by(name, region, source) %>%
  slice_max(final_score, with_ties = FALSE) %>%
  ungroup()

# Verification message
message("Number of best-case scenarios identified: ", nrow(best_by_group), "\n")
print(best_by_group$stock)

#----------------------------------------
# Plot the summary of best-case scenarios
#----------------------------------------
p6 <- best_by_group %>%
  dplyr::count(source, bk_method, r_method) %>%
  ggplot(aes(x = interaction(bk_method, r_method,sep = " - "), y = n, fill = source)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.85) +
  labs(
    x = "",
    y = "Number of best scenarios",
    fill = ""
  ) +
  scale_fill_viridis_d()+
  theme_classic(base_size = 15) %+replace%
  theme(
    strip.background = element_blank(), 
    plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"), # Tight plot margins
    axis.text.y =element_text(size = 16),
    axis.text.x =element_text(size = 16,angle = 45, vjust =1, hjust = 1),
    strip.text.x = element_text(
      margin = margin(b = 1),  # Bottom margin for facet labels
      size = 16
    ),
    legend.box.margin = margin(t = -40), # Move legend closer
    legend.text = element_text(size=16),
    legend.spacing.y = unit(0.1, "cm"),  # Reduce spacing between legend rows
    legend.position = "bottom"           # Place legend at bottom
  )

p_comb<-p5+p6
p_comb

# saving...
ggsave("best_models_metrics.png", plot = p_comb, device = "png",
                              width = 30, height = 16, units = "cm")

#-----------------------------------------
#Organizing best_by_group final results
#-----------------------------------------

# ---- Creating the summary table ----#
best_summary_table <- best_by_group %>%
  mutate(
    # Clear names from Stocks (remove everything after "_Base")
    stock = str_remove(stock, "_(Freire|Silva).*"),
    
    # Format ranges
    bk_range   = glue("{round(postbklw,2)}–{round(postbkup,2)}"),
    r_range    = glue("{round(postrlw,2)}–{round(postrup,2)}"),
    k_range    = glue("{round(postklw,1)}–{round(postkup,1)}"),
    msy_range  = glue("{round(postmsylw,2)}–{round(postmsyup,2)}"),
    
    # Combine priors and posteriors
    bk_combined  = glue("{round(priorbklw,2)}–{round(priorbkup,2)} / {bk_range}"),
    r_combined   = glue("{round(priorrlw,2)}–{round(priorrup,2)} / {r_range}"),
    k_combined   = glue("{round(priorklw,0)}–{round(priorkup,0)} / {k_range}"),
    msy_combined = glue("{round(priormsylw,2)}–{round(priormsyup,2)} / {msy_range}")
  ) %>%
  transmute(
    `Catch source` = source,
    Stock = stock,
    `r Method` = r_method,
    `B/k Method` = bk_method,
    `B/kprior / B/kpost` = bk_combined,
    `rprior / rpost` = r_combined,
    `kprior / kpost` = k_combined,
    `MSYprior / MSYpost` = msy_combined,
    RUr = round(RUr, 2),
    OVLr = round(OVLr, 2),
    `VR` = round(viab_rate, 2)*100
  )

# View result
print(best_summary_table, n = 10)

#writing the final best_by_group data frame
writexl::write_xlsx(best_summary_table,"best_summary_table.xlsx")
# write.table(best_summary_table,"best_summary_table.csv",append =FALSE,dec=".",sep = ",",
#             row.names = FALSE)


# ==============================================================
# Management table (Ct last years, B/Bmsy and F/Fmsy and Status)
# ==============================================================
#best scenarios
best_by_group <- best_by_group %>%
  mutate(stock_base = paste(name, region, sep = "_"))
#ct data
ct_out <- ct_out %>%
  mutate(stock_base = sub("^([^_]+_[^_]+).*", "\\1", Stock))
# B/Bmsy and F/Fmsy data for 2015
bbmsy_ffmsy_series_1950_2015<-bio_out %>%
  filter(stock %in% c(unique(best_by_group$stock))) %>%
  mutate(stock_base = paste(name, region, sep = "_"))
# B/Bmsy and F/Fmsy data for 2025
bbmsy_ffmsy_series_2016_2025<-bio_out %>%
  filter(grepl("_Projection", stock)) %>%
  mutate(stock_base = paste(name, region, sep = "_"))


#catches
ct_hist_summary <- ct_out %>%
  dplyr::filter(yr >= 2010 & yr <= 2015) %>%
  dplyr::group_by(stock_base) %>%
  dplyr::summarise(
    ct_2010_2015_mean =round(mean(ct, na.rm = TRUE),1)/1000,
    ct_2015_mean      =round(mean(ct[yr == 2015], na.rm = TRUE),1)/1000
  )

#Catch projected
ct_proj_summary <- ct_out %>%
  dplyr::filter(yr >= 2016 & yr <= 2025) %>%
  dplyr::group_by(stock_base) %>%
  dplyr::summarise(
    ct_proj_2016_2025_mean = round(mean(ct, na.rm = TRUE),1)/1000,
    ct_proj_2025           = round(mean(ct[yr == 2025], na.rm = TRUE),1)/1000
  )

# mean MSY 
msy_mean<- best_by_group %>%
  dplyr::group_by(stock_base) %>%
  dplyr::summarise(
    msy_mean = round(mean(postmsy, na.rm = TRUE),1) )

# B/BMSY  and F/Fmsy (2015)
bbmsy_ffmsy_summary_2015 <- bbmsy_ffmsy_series_1950_2015 %>%
  dplyr::filter(yr >= 2015) %>%
  dplyr::group_by(stock_base) %>%
  dplyr::summarise(
    bbmsy_2015_mean = round(mean(B.Bmsy, na.rm = TRUE),1),
    ffmsy_2015_mean = round(mean(F.Fmsy, na.rm = TRUE),1)
  )

# B/BMSY  and F/Fmsy (2025)
bbmsy_ffmsy_summary_2025 <- bbmsy_ffmsy_series_2016_2025 %>%
  dplyr::filter(yr >= 2025) %>%
  dplyr::group_by(stock_base) %>%
  dplyr::summarise(
    bbmsy_2025_mean = round(mean(B.Bmsy, na.rm = TRUE),1),
    ffmsy_2025_mean = round(mean(F.Fmsy, na.rm = TRUE),1)
  )

# Put it all together in a final table
management_table <- ct_hist_summary %>%
  full_join(ct_proj_summary, by = "stock_base") %>%
  full_join(msy_mean, by = "stock_base") %>%
  full_join(bbmsy_ffmsy_summary_2015,   by = "stock_base") %>%
  full_join(bbmsy_ffmsy_summary_2025,   by = "stock_base") %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))

# Status classification
classify_status <- function(Fv, Bv) {
  
  if (is.na(Fv) || is.na(Bv)) return("Data deficient")
  
  # fully fished 
  if (Fv == 1 && Bv == 1) return("Fully fished")
  
  # Overfishing scenarios
  if (Fv > 1 && Bv >= 1) return("Overfishing")
  if (Fv > 1 && Bv < 1)  return("Overfishing and overfished")
  if (Fv < 1 && Bv < 1)  return("Overfished")
  
  #underfishing
  if (Fv < 1 && Bv > 1)  return("Underfished")
  
  #If one of them equals 1, use the other indicator:
  if (Fv == 1 && Bv > 1) return("Underfished")
  if (Fv == 1 && Bv < 1) return("Overfished")
  if (Bv == 1 && Fv > 1) return("Overfishing")
  if (Bv == 1 && Fv < 1) return("Underfished")
  
  # safe fallback 
  return("Undefined")
}

management_table <- management_table %>%
  mutate(
    Status2015 = mapply(classify_status, ffmsy_2015_mean, bbmsy_2015_mean),
    Status2025 = mapply(classify_status, ffmsy_2025_mean, bbmsy_2025_mean)
  )
management_table

#writing the final management_table data frame
writexl::write_xlsx(management_table,"management_table.xlsx")


#----------------------------------------------------------------------
# Projections scenarios (2016-2025) and base case scenarios (1950-2015)
#----------------------------------------------------------------------
 
# All series tested (background) - Base and Projection separated
all_base_series <- bio_out %>% filter(grepl("_Base", stock))
all_proj_series <- bio_out %>% filter(grepl("_Projection", stock))

# Ensure that best_by_group exists and contains the best scenarios (Base)
# best_by_group should have columns: stock, name, region, source
# Filter only the best *Base* (in theory best_by_group was obtained from the Base)
best_base_stocks <- best_by_group %>% filter(grepl("_Base", stock)) %>% pull(stock)

#Extract trajectories corresponding to the best-case scenarios. Base
best_base_trajs <- bio_out %>%
  filter(stock %in% best_base_stocks) %>%
  # keep B.Bmsy, F.Fmsy, name, region, source, yr, stock
  select(stock, source, name, region, yr, B.Bmsy, F.Fmsy)

#Now calculate the average between the best (typically Freire + Silva) by name/region/year
best_base_summary <- best_base_trajs %>%
  dplyr::group_by(name, region, yr) %>%
  dplyr::summarise(
    B_mean = mean(B.Bmsy, na.rm = TRUE),
    B_lcl  = quantile(B.Bmsy, 0.025, na.rm = TRUE),
    B_ucl  = quantile(B.Bmsy, 0.975, na.rm = TRUE),
    F_mean = mean(F.Fmsy, na.rm = TRUE),
    F_lcl  = quantile(F.Fmsy, 0.025, na.rm = TRUE),
    F_ucl  = quantile(F.Fmsy, 0.975, na.rm = TRUE),
    n_sources = n_distinct(source), # how many sources contributed
    .groups = "drop"
  )

# ----------------------------------------------------------
# Projections: take the Projection trajectories and
# calculate the average between the sources (Freire & Silva)
# ----------------------------------------------------------
# Project stocks: name_source_Projection
proj_series <- bio_out %>% filter(grepl("_Projection", stock))

# List of projection stocks by source for each name/region (useful to check)
# Now calculate the average between projections by name/region/year
# Summary of projections (average between sources)
# -----------------------------------------------------
# Projections: take the trajectories from Projection and
# calculate the average between the sources (Freire & Silva)
# ------------------------------------------------------
proj_summary <- bio_out %>%
  filter(grepl("_Projection", stock)) %>%
  filter(yr >= 2016) %>%  # real projections only from 2016 onwards
  dplyr::group_by(name, region, yr) %>%
  dplyr::summarise(
    B_mean = mean(B.Bmsy, na.rm = TRUE),
    B_lcl  = quantile(B.Bmsy, 0.025, na.rm = TRUE),
    B_ucl  = quantile(B.Bmsy, 0.975, na.rm = TRUE),
    F_mean = mean(F.Fmsy, na.rm = TRUE),
    F_lcl  = quantile(F.Fmsy, 0.025, na.rm = TRUE),
    F_ucl  = quantile(F.Fmsy, 0.975, na.rm = TRUE),
    n_sources = n_distinct(source),
    .groups = "drop"
  )

# -------------------------------------------------------
# Add the 2015 point from the best-case scenarios (Base)
# -------------------------------------------------------
# Extract the 2015 lines from the best-case historical scenario
base_2015 <- best_base_summary %>%
  filter(yr == 2015) %>%
  mutate(n_sources = NA_integer_)  # just to maintain column consistency

# Combine with the projections
proj_summary <- bind_rows(base_2015, proj_summary) %>%
  arrange(name, region, yr)

# -----------------------------------------------------
# Sanity checks - see how many sources contributed
# ----------------------------------------------------
# Which name/region has fewer than 2 sources among the top ones?
check_best_n <- best_base_trajs %>%
  group_by(name, region) %>%
  summarise(n_sources = n_distinct(source), .groups = "drop") %>%
  filter(n_sources < 2)

if (nrow(check_best_n) > 0) {
  message("Note: some name-regions have < 2 sources in the best ones (maybe only 1):")
  print(check_best_n)
}

# ---------------------------------------------------------------
# Plot: all clear series + averages of the best + projections
# ---------------------------------------------------------------
# Create combined data frames for legend mapping
best_base_summary_long <- best_base_summary %>%
  tidyr::pivot_longer(
    cols = c(B_mean, F_mean),
    names_to = "metric",
    values_to = "value"
  )

proj_summary_long <- proj_summary %>%
  tidyr::pivot_longer(
    cols = c(B_mean, F_mean),
    names_to = "metric",
    values_to = "value"
  )
#----------------
# Final plot--> Base case scenarios + Projections
#----------------
p7 <- ggplot() +
  # Background with all series
  geom_line(data = all_base_series,
            aes(x = yr, y = B.Bmsy, group = interaction(stock, source)),
            color = alpha("#56B4E9", 0.20)) +
  geom_line(data = all_base_series,
            aes(x = yr, y = F.Fmsy, group = interaction(stock, source)),
            color = alpha("#D55E00", 0.20)) +
  geom_line(data = all_proj_series,
            aes(x = yr, y = B.Bmsy, group = interaction(stock, source)),
            color = alpha("#56B4E9", 0.12), linetype = "dotted") +
  geom_line(data = all_proj_series,
            aes(x = yr, y = F.Fmsy, group = interaction(stock, source)),
            color = alpha("#D55E00", 0.12), linetype = "dotted") +
  # Main lines with legend
  geom_line(data = best_base_summary_long,
            aes(x = yr, y = value, color = metric),
            linewidth = 1) +
  geom_line(data = proj_summary_long,
            aes(x = yr, y = value, color = metric),
            linewidth = 1, linetype = "dashed") +
  # Intervals
  geom_ribbon(data = best_base_summary,
              aes(x = yr, ymin = B_lcl, ymax = B_ucl),
              fill = alpha("#56B4E9", 0.25)) +
  geom_ribbon(data = best_base_summary,
              aes(x = yr, ymin = F_lcl, ymax = F_ucl),
              fill = alpha("#D55E00", 0.25)) +
  geom_ribbon(data = proj_summary,
              aes(x = yr, ymin = B_lcl, ymax = B_ucl),
              fill = alpha("#56B4E9", 0.15)) +
  geom_ribbon(data = proj_summary,
              aes(x = yr, ymin = F_lcl, ymax = F_ucl),
              fill = alpha("#D55E00", 0.15)) +
  geom_vline(xintercept = 2015, linetype = "dotted", color = "grey40") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
  facet_wrap(region ~ name, scales = "free_y") +
  labs(
    x = "Year",
    y = expression(B/B[MSY] ~ " and " ~ F/F[MSY]),
    color = NULL,   
    fill = NULL
  ) +
  scale_color_manual(
    values = c("B_mean" = "#56B4E9", "F_mean" = "#D55E00"),
    labels = c(expression(B/B[MSY]), expression(F/F[MSY]))
  ) +
  theme_classic(base_size = 11) %+replace%
  theme(
    strip.background = element_blank(), 
    plot.margin = unit(c(0.05, 0.05, 0.05, 0.05), "mm"), # Tight plot margins
    axis.text.y =element_text(size = 11),
    axis.text.x =element_text(size = 11),
    strip.text.x = element_text(
      margin = margin(b = 1),  # Bottom margin for facet labels
      size = 11
    ),
    legend.box.margin = margin(t = -10), # Move legend closer
    legend.text = element_text(size=11),
    legend.spacing.y = unit(0.1, "cm"),  # Reduce spacing between legend rows
    legend.position = "bottom"           # Place legend at bottom
  )

p7

# saving...
ggsave("Biomass_Projections_bestseries.png", 
       plot = p7, device = "png",  units = "cm", width = 35, height = 18)
#--------------------------------------------------------------------------




# ================================================================
# Full projection pipeline (using posterior r,k draws from rk_out)
# ================================================================
# library(dplyr)
# library(tidyr)
# library(purrr)
# library(ggplot2)
# library(scales)
# 
# # ---------------------------
# # User data (must exist)
# # ---------------------------
# # best_by_group : best scenarios (one per name/region/source)
# # rk_out        : posterior draws for r and K (with stock, name, region, source, postr, postk)
# # cmsy_out      : CMSY output (contains postmsy, ffmsy, bbmsy, etc.)
# # bio_out       : historical trajectories (B/Bmsy and F/Fmsy)
# # cdat          : catch table including *_Projection stocks
# 
# # ---------------------------
# # Projection parameters
# # ---------------------------
# N_DRAWS <- 2000
# PROC_SD <- 0.12
# YR_PROJ <- 2016:2025
# EPS <- 1e-4
# LOWP_THRESHOLD <- 0.25
# 
# # ---------------------------
# # Helper: get posterior r,k for a stock
# # ---------------------------
# get_rk_posteriors <- function(best_stock, rk_out, n_draws = 2000) {
#   rk_sub <- rk_out %>% filter(stock == best_stock)
#   if (nrow(rk_sub) == 0) stop("No posterior r,k found for ", best_stock)
#   if (nrow(rk_sub) > n_draws) rk_sub <- rk_sub %>% sample_n(n_draws)
#   list(r = rk_sub$postr, k = rk_sub$postk)
# }
# 
# # ---------------------------
# # Helper: get B/Bmsy at 2015 (final historical year)
# # ---------------------------
# get_p2015_draws <- function(best_stock, bio_out, cmsy_out, min_draws = 200) {
#   b_bmsy_vals <- bio_out %>%
#     filter(stock == best_stock, yr == 2015) %>%
#     pull(B.Bmsy)
#   
#   b_bmsy_vals <- b_bmsy_vals[is.finite(b_bmsy_vals) & !is.na(b_bmsy_vals)]
#   if (length(b_bmsy_vals) < min_draws) {
#     warning("Low number of P2015 draws for ", best_stock, " — replicating available values.")
#     b_bmsy_vals <- rep(b_bmsy_vals, length.out = min_draws)
#   }
#   
#   # converter para B/K, assumindo Bmsy = K/2
#   p2015_vals <- b_bmsy_vals / 2
#   p2015_vals
# }
# 
# # ---------------------------
# # Helper: get Bmsy and Fmsy values for a stock
# # ---------------------------
# get_ref_metrics <- function(best_stock, cmsy_out) {
#   ref <- cmsy_out %>% filter(stock == best_stock)
#   if (nrow(ref) == 0) stop("No reference metrics found for ", best_stock)
#   list(Bmsy = ref$bmsy[1], Fmsy = ref$fmsy[1])
# }
# 
# # ---------------------------
# # Forward projection (Schaefer-like dynamics)
# # Returns trajectories already normalized by B/Bmsy and F/Fmsy
# # ---------------------------
# forward_project_schaefer <- function(r_draws, k_draws, p2015_draws, 
#                                      C_proj, yrs_proj, 
#                                      Bmsy, Fmsy,
#                                      n_sims = 2000, proc_sd = 0.12,
#                                      eps = 1e-4, lowP_threshold = 0.25, seed = NULL) {
#   if (!is.null(seed)) set.seed(seed)
#   sample_vec <- function(x, n) if(length(x) >= n) sample(x, n, FALSE) else sample(x, n, TRUE)
#   r_samp <- sample_vec(r_draws, n_sims)
#   k_samp <- sample_vec(k_draws, n_sims)
#   p0_samp <- sample_vec(p2015_draws, n_sims)
#   
#   n_proj <- length(yrs_proj)
#   P_mat <- matrix(NA, n_sims, n_proj)
#   B_mat <- matrix(NA, n_sims, n_proj)
#   F_mat <- matrix(NA, n_sims, n_proj)
#   
#   for (i in seq_len(n_sims)) {
#     r_i <- r_samp[i]; k_i <- k_samp[i]; P_prev <- max(p0_samp[i], eps)
#     for (t in seq_len(n_proj)) {
#       C_t <- C_proj[t]
#       det_next <- if (P_prev > lowP_threshold)
#         P_prev + r_i * P_prev * (1 - P_prev) - (C_t / k_i)
#       else
#         P_prev + 4 * P_prev * r_i * P_prev * (1 - P_prev) - (C_t / k_i)
#       det_next <- max(det_next, eps)
#       P_next <- exp(rnorm(1, mean = log(det_next), sd = proc_sd))
#       P_next <- max(P_next, eps)
#       
#       B_now <- P_next * k_i
#       E <- ifelse(B_now > 0, C_t / B_now, NA_real_)
#       F_now <- ifelse(!is.na(E), -log(pmax(1 - E, 1e-6)), NA_real_)
#       
#       P_mat[i, t] <- P_next
#       B_mat[i, t] <- B_now / Bmsy
#       F_mat[i, t] <- F_now / Fmsy
#       P_prev <- P_next
#     }
#   }
#   
#   df_summary <- map_dfr(seq_len(n_proj), function(j) {
#     data.frame(
#       yr = yrs_proj[j],
#       B_mean = mean(B_mat[, j], na.rm = TRUE),
#       B_lcl  = quantile(B_mat[, j], 0.025, na.rm = TRUE),
#       B_ucl  = quantile(B_mat[, j], 0.975, na.rm = TRUE),
#       F_mean = mean(F_mat[, j], na.rm = TRUE),
#       F_lcl  = quantile(F_mat[, j], 0.025, na.rm = TRUE),
#       F_ucl  = quantile(F_mat[, j], 0.975, na.rm = TRUE)
#     )
#   })
#   
#   list(summary = df_summary, B_mat = B_mat, F_mat = F_mat)
# }
# 
# # ---------------------------
# # Prepare projection catch data
# # ---------------------------
# ct_proj_df <- cdat %>%
#   dplyr::rename(stock = Stock) %>%
#   filter(grepl("_Projection", stock)) %>%
#   arrange(stock, yr)
# 
# get_C_proj_for_stock <- function(proj_stock_name, yrs_proj = YR_PROJ, ct_proj_df, bw = 3) {
#   tmp <- ct_proj_df %>% 
#     dplyr::filter(stock == proj_stock_name, yr %in% yrs_proj) %>% 
#     dplyr::arrange(yr)
#   
#   if (nrow(tmp) == 0) stop("No projection catches found for ", proj_stock_name)
#   
#   yrs <- tmp$yr
#   ct.raw <- tmp$ct / 1000
#   
#   # Aplicar suavização apenas sobre os anos de projeção
#   ct.smooth <- ksmooth( x = yrs,y = ct.raw,  kernel = "normal", n.points = length(yrs),  bandwidth = bw)$y
#   
#   # Garante que o vetor de capturas tenha o mesmo comprimento e ordem que yrs_proj
#   df_proj <- data.frame(yr = yrs, ct = ct.smooth) %>%
#     right_join(data.frame(yr = yrs_proj), by = "yr") %>%
#     arrange(yr)
#   
#   list(ct = df_proj$ct, yrs = df_proj$yr)
# }
# 
# # ---------------------------
# # Main loop over best scenarios
# # ---------------------------
# proj_results_list <- list()
# 
# for (i in seq_len(nrow(best_by_group))) {
#   row <- best_by_group[i, ]
#   best_stock <- as.character(row$stock)
#   stock_prefix <- sub("_Base.*", "", best_stock)
#   message(">>> Running projection for ", best_stock)
#   
#   # Find matching projection stock
#   proj_stock_name <- paste0(stock_prefix, "_Projection")
#   if (!proj_stock_name %in% ct_proj_df$stock) next
#   
#   cinfo_proj <- get_C_proj_for_stock(proj_stock_name, yrs_proj = YR_PROJ, ct_proj_df)
#   
#   # Get posterior r,k, B/Bmsy(2015), and reference metrics
#   rk_vals <- get_rk_posteriors(best_stock, rk_out, n_draws = N_DRAWS)
#   p2015_vals <- get_p2015_draws(best_stock, bio_out, min_draws = 200)
#   ref_vals <- get_ref_metrics(best_stock, cmsy_out)
#   
#   # Run projection
#   proj_res <- forward_project_schaefer(
#     r_draws = rk_vals$r, k_draws = rk_vals$k,
#     p2015_draws = p2015_vals, C_proj = cinfo_proj$ct,
#     yrs_proj = cinfo_proj$yrs, Bmsy = ref_vals$Bmsy,
#     Fmsy = ref_vals$Fmsy, n_sims = N_DRAWS, proc_sd = PROC_SD, eps = EPS
#   )
#   
#   proj_results_list[[best_stock]] <- list(
#     best_stock = best_stock,
#     proj_stock = proj_stock_name,
#     name = row$name,
#     region = row$region,
#     source = row$source,
#     summary = proj_res$summary
#   )
# }
# 
# # ---------------------------
# # Combine summaries
# # ---------------------------
# proj_summaries <- map_dfr(proj_results_list, \(x) {
#   df <- x$summary
#   df$name <- x$name
#   df$region <- x$region
#   df$source <- x$source
#   df$best_stock <- x$best_stock
#   df
# })
# 
# # Aggregate across sources
# agg_proj_by_name_region <- proj_summaries %>%
#   dplyr::group_by(name, region, yr) %>%
#   dplyr::summarise(
#     B_mean = mean(B_mean, na.rm = TRUE),
#     B_lcl = quantile(B_mean, 0.025, na.rm = TRUE),
#     B_ucl = quantile(B_mean, 0.975, na.rm = TRUE),
#     F_mean = mean(F_mean, na.rm = TRUE),
#     F_lcl = quantile(F_mean, 0.025, na.rm = TRUE),
#     F_ucl = quantile(F_mean, 0.975, na.rm = TRUE),
#     .groups = "drop"
#   )
# 
# # Save results
# write.csv(proj_summaries, "proj_summaries_per_beststock.csv", row.names = FALSE)
# write.csv(agg_proj_by_name_region, "proj_summaries_aggregated_name_region.csv", row.names = FALSE)
# 
# message("Done: projections completed for ", length(proj_results_list), " stocks.")
# 
# # ---------------------------
# # Example plots (aggregated mean projections + historical best averages)
# # We'll also compute best_base_summary (average of best Base trajectories from bio_out)
# # ---------------------------
# 
# # =====================================================================
# # Painel combinado: histórico (Base, CMSY) + projeções (Schaefer)
# # =====================================================================
# 
# library(dplyr)
# library(ggplot2)
# library(scales)
# 
# # ------------------------------------------------------------------
# # 1. Séries base (1950–2015) - Histórico
# # ------------------------------------------------------------------
# 
# # Todas as séries testadas (para fundo visual)
# all_base_series <- bio_out %>% filter(grepl("_Base", stock))
# 
# # Filtrar apenas os melhores cenários base
# best_base_stocks <- best_by_group %>%
#   filter(grepl("_Base", stock)) %>%
#   pull(stock)
# 
# # Extrair trajetórias dos melhores cenários base
# best_base_trajs <- bio_out %>%
#   filter(stock %in% best_base_stocks) %>%
#   select(stock, source, name, region, yr, B.Bmsy, F.Fmsy)
# 
# # Resumir entre fontes (por name/region/yr)
# best_base_summary <- best_base_trajs %>%
#   dplyr::group_by(name, region, yr) %>%
#   dplyr::summarise(
#     B_mean = mean(B.Bmsy, na.rm = TRUE),
#     B_lcl  = quantile(B.Bmsy, 0.025, na.rm = TRUE),
#     B_ucl  = quantile(B.Bmsy, 0.975, na.rm = TRUE),
#     F_mean = mean(F.Fmsy, na.rm = TRUE),
#     F_lcl  = quantile(F.Fmsy, 0.025, na.rm = TRUE),
#     F_ucl  = quantile(F.Fmsy, 0.975, na.rm = TRUE),
#     .groups = "drop"
#   )
# 
# # ------------------------------------------------------------------
# # 2. Séries de projeção (Schaefer forward simulation)
# # ------------------------------------------------------------------
# # Já foram calculadas via forward_project_schaefer e estão salvas em:
# # proj_summaries (por stock) e agg_proj_by_name_region (agregado)
# 
# proj_summary <- agg_proj_by_name_region
# 
# # ------------------------------------------------------------------
# # 3. Plotar painel histórico + projeções
# # ------------------------------------------------------------------
# 
# p_combined <- ggplot() +
#   # Fundo: todas as séries Base (linhas claras)
#   geom_line(
#     data = all_base_series,
#     aes(x = yr, y = B.Bmsy, group = interaction(stock, source)),
#     color = alpha("#56B4E9", 0.20)
#   ) +
#   geom_line(
#     data = all_base_series,
#     aes(x = yr, y = F.Fmsy, group = interaction(stock, source)),
#     color = alpha("#D55E00", 0.20)
#   ) +
#   
#   # Ribbon + média dos melhores (Base)
#   geom_ribbon(
#     data = best_base_summary,
#     aes(x = yr, ymin = B_lcl, ymax = B_ucl),
#     fill = alpha("#56B4E9", 0.25)
#   ) +
#   geom_ribbon(
#     data = best_base_summary,
#     aes(x = yr, ymin = F_lcl, ymax = F_ucl),
#     fill = alpha("#D55E00", 0.25)
#   ) +
#   geom_line(
#     data = best_base_summary,
#     aes(x = yr, y = B_mean),
#     color = "#56B4E9", size = 1
#   ) +
#   geom_line(
#     data = best_base_summary,
#     aes(x = yr, y = F_mean),
#     color = "#D55E00", size = 1
#   ) +
#   
#   # Ribbon + média das projeções (Schaefer)
#   geom_ribbon(
#     data = proj_summary,
#     aes(x = yr, ymin = B_lcl, ymax = B_ucl),
#     fill = alpha("#56B4E9", 0.15)
#   ) +
#   geom_ribbon(
#     data = proj_summary,
#     aes(x = yr, ymin = F_lcl, ymax = F_ucl),
#     fill = alpha("#D55E00", 0.15)
#   ) +
#   geom_line(
#     data = proj_summary,
#     aes(x = yr, y = B_mean),
#     color = "#56B4E9", size = 1, linetype = "dashed"
#   ) +
#   geom_line(
#     data = proj_summary,
#     aes(x = yr, y = F_mean),
#     color = "#D55E00", size = 1, linetype = "dashed"
#   ) +
#   
#   # Linhas de referência
#   geom_vline(xintercept = 2015, linetype = "dotted", color = "grey40") +
#   geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
#   
#   # Facetas e temas
#   facet_wrap(region ~ name, scales = "free_y") +
#   labs(
#     x = "Year",
#     y = expression(B/B[MSY] ~ " and " ~ F/F[MSY]),
#     title = "Historical (Base, solid) and Projected (Schaefer, dashed) Trajectories"
#   ) +
#   theme_classic(base_size = 12) +
#   theme(
#     strip.background = element_blank(),
#     axis.text.x = element_text(angle = 45, hjust = 1),
#     panel.grid = element_blank()
#   )
# 
# # Visualizar
# print(p_combined)
# 
# # Salvar
# ggsave("Biomass_Exploitation_SchaeferProjections.png",
#        plot = p_combined,
#        device = "png", width = 34, height = 20, units = "cm")
# 
# 
# message("Done: projections executed for ", length(proj_results_list), " best scenarios.")
# message("Saved proj_summaries_per_beststock.csv and proj_summaries_aggregated_name_region.csv and example plots.")


# -------------------------------------
# KOBE plot building
# -------------------------------------
# {#kobe plot data preparation
# #graphical parameters
# png("Kobe.png",width=35,height=22,units="cm",
#   res=800,
#   antialias="cleartype")
# par(mfrow=c(3,4),mar=c(1.9,2,0.1,0),
#   oma=c(1.3,2.11,0.5,0.5),xpd=F,
#   mgp=c(1.1,0.55,0), cex.lab=0.9,cex.axis=0.7, bty="L",las=1,
#   col.axis='gray30',col.lab='gray10')
# 
# #subseting the bioout for the best runs (lowest ppvrbk)
# source=unique(ppintervalsout$source)
# category= unique(ppintervalsout$category)
# region= unique(ppintervalsout$region)
# bestruns<- data.frame(stock=NULL)
# 
# for (i in source) { #loop to find the lowest ppvrbk for each source
#   for (j in category) {
#     for (k in region) {
#       
#       bst<-  ppintervalsout$stock[ppintervalsout$source==i & 
#           ppintervalsout$category==j & 
#           ppintervalsout$region==k ] [ which.min(ppintervalsout$ppvrbk[ppintervalsout$source==i & 
#               ppintervalsout$category==j & 
#               ppintervalsout$region==k ] )]
#       best<-data.frame(stock=bst)
#       bestruns<-rbind(bestruns,best)
#     }
#   }
# }
# 
# # Subsetting kobeout and bioout for the best runs
# kobeout_sub <- kobeout[kobeout$stock %in% bestruns$stock, ]
# bioout_sub <- bioout[bioout$stock %in% bestruns$stock, ]
# 
# #vectors of categories and regions
# category= unique(kobeout_sub$category)
# region= unique(kobeout_sub$region)
# for (i in category) { #loop through them
#   for (j in region) {
#     
#     start.yr= unique(bioout_sub$start.yr[bioout_sub$category==i & bioout_sub$region==j])
#     int.yr= unique(bioout_sub$int.yr[bioout_sub$category==i & bioout_sub$region==j])
#     end.yr= unique(bioout_sub$end.yr[bioout_sub$category==i & bioout_sub$region==j])
#     nyr= unique(bioout_sub$nyr[bioout_sub$category==i & bioout_sub$region==j])
#     B.Bmsy= data.frame(yr=bioout_sub$yr[bioout_sub$category==i & bioout_sub$region==j],
#       b=bioout_sub$B.Bmsy[bioout_sub$category==i & bioout_sub$region==j],
#       source=bioout_sub$source[bioout_sub$category==i & bioout_sub$region==j])
#     F.Fmsy= data.frame(yr=bioout_sub$yr[bioout_sub$category==i & bioout_sub$region==j],
#       f=bioout_sub$F.Fmsy[bioout_sub$category==i & bioout_sub$region==j],
#       source=bioout_sub$source[bioout_sub$category==i & bioout_sub$region==j])
#     
#     #f/fmsy and b/bmsy in the last year (2015) for all sources
#     x.F_Fmsy = kobeout_sub$x.F_Fmsy[kobeout_sub$category==i & kobeout_sub$region==j]
#     y.b_bmsy = kobeout_sub$y.b_bmsy[kobeout_sub$category==i & kobeout_sub$region==j]
#     
#     # Next iteration if the category x region combination has no data
#     if (length(x.F_Fmsy) == 0 & length(y.b_bmsy) == 0) {
#       next
#     }
#     
#     max.y    <- max(c(F.Fmsy$f,x.F_Fmsy))
#     max.x    <- max(c(B.Bmsy$b,y.b_bmsy))
#     min.x    <- min(c(B.Bmsy$b,y.b_bmsy))
#     
#     # Create plot
#     plot(100, 100, type="b", xlim=c(min.x-0.08, max.x+0.05), ylim=c(0, max.y), lty=3, xlab="", ylab="",
#       bty="l", cex.main = 2, cex.lab = 1.35, cex.axis = 1.35, xaxs = "i", yaxs="i", col="white")
#     
#     c1 <- c(-1, 100)
#     c2 <- c(1, 1)
#     
#     # extract interval information from ci2d object
#     # and fill areas using the polygon function
#     zb2 = c(0, 1)
#     zf2  = c(1, 100)
#     zb1 = c(1, 100)
#     zf1  = c(0, 1)
#     polygon(c(zb1, rev(zb1)), c(0, 0, 1, 1), col="green", border=0)
#     polygon(c(zb2, rev(zb2)), c(0, 0, 1, 1), col="yellow", border=0)
#     polygon(c(1, 100, 100, 1), c(1, 1, 100, 100), col="orange", border=0)
#     polygon(c(0, 1, 1, 0), c(1, 1, 100, 100), col="red", border=0)
#     
#     sources = unique(B.Bmsy$source)
#     sources <- sources[order(match(sources, c("Silva")))] #ordering by Silva
#     for (k in sources) {
#       col = if (k == "Silva") {
#         coll = "gray10"
#         colp = rgb(red=0.2, green=0.2, blue=0.25, alpha = 0.5)
#       } else if (k == "Freire") {
#         coll = 'gray70'
#         colp = rgb(red=0.6, green=0.6, blue=0.6, alpha = 0.5)
#       }
#       
#       # cloud of points F/Fmsy B/Bmsy for the last year
#       points(kobeout_sub$y.b_bmsy[kobeout_sub$category == i & kobeout_sub$region == j & kobeout_sub$source == k],
#         kobeout_sub$x.F_Fmsy[kobeout_sub$category == i & kobeout_sub$region == j & kobeout_sub$source == k],
#         pch=20, col=colp) 
#       
#       lines(B.Bmsy$b[B.Bmsy$source == k], F.Fmsy$f[F.Fmsy$source == k], lwd=2, col=coll)
#       
#       # f/fmsy and b/bmsy points in the first year for each source
#       points(B.Bmsy$b[B.Bmsy$source == k][1], F.Fmsy$f[F.Fmsy$source == k][1], col=coll, pch=19, bg='white', cex=1.5)
#     }
#     
#     # Draw the final year points last to ensure they are not covered
#     for (k in sources) {
#       col = if (k == "Silva") {
#         coll = "gray10"
#         colp = rgb(red=0.2, green=0.2, blue=0.25, alpha = 0.5)
#       } else if (k == "Freire"){
#         coll = 'gray70'
#         colp = rgb(red=0.6, green=0.6, blue=0.6, alpha = 0.5)
#       }
#       
#       
#       # f/fmsy and b/bmsy points in the final year for each source
#       points(B.Bmsy$b[B.Bmsy$source == k][length(B.Bmsy$b[B.Bmsy$source == k])], 
#              F.Fmsy$f[F.Fmsy$source == k][length(B.Bmsy$b[B.Bmsy$source == k])], 
#           col=coll, pch=17, bg='white', cex=1.5)
#       lines(B.Bmsy$b[B.Bmsy$source == k], F.Fmsy$f[F.Fmsy$source == k], lwd=2, col=coll)
#     }
#     
#     lines(c1, c2, lty=3, lwd=0.7)
#     lines(c2, c1, lty=3, lwd=0.7)
#     
#     # Get Propability
#     Pr.green = sum(ifelse(y.b_bmsy > 1 & x.F_Fmsy < 1, 1, 0)) / length(y.b_bmsy) * 100
#     Pr.red = sum(ifelse(y.b_bmsy < 1 & x.F_Fmsy > 1, 1, 0)) / length(y.b_bmsy) * 100
#     Pr.yellow = sum(ifelse(y.b_bmsy < 1 & x.F_Fmsy < 1, 1, 0)) / length(y.b_bmsy) * 100
#     Pr.orange = sum(ifelse(y.b_bmsy > 1 & x.F_Fmsy > 1, 1, 0)) / length(y.b_bmsy) * 100
#     
#     if (length(sources) > 1) {
#       legend('topright',
#         c(paste(i, j, sep="-"), paste(start.yr), paste(end.yr), paste(sources),
#           paste0(round(c(Pr.red, Pr.yellow, Pr.orange, Pr.green), 1), "%")),
#         lty=c(NA, 1, 1, 1, 1, rep(-1, 4)),
#         pch=c(NA, 19, 17, NA, NA, rep(22, 4)),
#         pt.bg=c("gray", rep("gray40", 2), "gray10", "gray60", "red", "yellow", "orange", "green"),
#         col=c("gray", rep("gray40", 2), "gray10", "gray60", rep(1, 4)), lwd=c(NA, rep(2.5, 4), rep(1.1, 4)),
#         cex=1.1, pt.cex=c(rep(1.3, 5), rep(2.2, 4)), bty="n", y.intersp = 1.)
#     }
#     
#     if (length(sources) == 1) {
#       legend('topright',
#         c(paste(i, j, sep="-"), paste(start.yr), paste(end.yr), paste(sources),
#           paste0(round(c(Pr.red, Pr.yellow, Pr.orange, Pr.green), 1), "%")),
#         lty=c(NA, 1, 1, 1, rep(-1, 4)),
#         pch=c(NA, 19, 17, NA, rep(22, 4)),
#         pt.bg=c("gray", rep("gray40", 2), "gray10", "red", "yellow", "orange", "green"),
#         col=c("gray", rep("gray40", 2), "gray10", rep(1, 4)), lwd=c(NA, rep(2.5, 3), rep(1.1, 4)),
#         cex=1.1, pt.cex=c(rep(1.3, 4), rep(2.2, 4)), bty="n", y.intersp = 1.)
#     }
#     
#     mtext(expression(B/B[MSY]), side=1, outer=T, line = 0.4, cex=1.3)
#     par(las=0)
#     mtext(expression(F/F[MSY]), side=2, outer=T, cex=1.3)
#    }
#  }
#  dev.off()
# }
#----------------------------------------------------------



##################################################################################################################
                                                 #End of assessments
##################################################################################################################
