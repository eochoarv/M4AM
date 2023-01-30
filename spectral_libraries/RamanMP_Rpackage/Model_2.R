library(randomForest)

rf_train = randomForest(polymer~ ., data_train, mtry = 5, ntree = 1000, importance = TRUE)

rf_pred_train = predict(rf_train, x_train, type = "class")
rf_pred_test = predict(rf_train, x_mat_tets, type = "class")
mean(rf_pred_train == as.factor(y_train)) 
mean(as.numeric(rf_pred_test) == y_test) 

x_mat_tets <- as.matrix(MPdatabase_inter_norm_pivot_w_polymers[,-c(1,2,3,1402)])
y_test <- as.character(MPdatabase_inter_norm_pivot_w_polymers[, 2])

y_test_2 = left_join(data.frame(polymer=as.character(y_test)), 
                     slopp_polymer_dict, by="polymer")$class

y_pred = predict(rf_train, x_mat_tets, type = "class")
mean(y_test_2 == y_pred, na.rm=T)

y_pred_prob <- predict(cvfit, newx = x_mat_tets, s = "lambda.min", type = "response")
y_pred_prob <- apply(y_pred_prob, MARGIN=1, FUN=max)

table(y_test_2, as.numeric(y_pred))

identif_spectra_polymer <- c()
identif_spectra_corr <- c()
i = 1
for (fn in MPdatabase_inter_norm_pivot_w_polymers$file_name) {
  identif_spectra <- spectra.corr(data_slopp_pivot, MPdatabase[, c('freq',fn)], t=0.5, normal='SNV')
  identif_spectra_polymer[i] = rownames(identif_spectra)[1]
  identif_spectra_corr[i] = identif_spectra[1]
  i = i + 1
}


corr_table <- left_join(data.frame(file_name_2=identif_spectra_polymer,
                                   corr=identif_spectra_corr,
                                   y_test),
                        as.data.frame(data_slopp %>% 
                                        select(file_name_2, polymer) %>% 
                                        group_by(file_name_2, polymer) %>% 
                                        summarise(n=n())), by='file_name_2') %>%
  select(polymer, y_test, corr) %>%
  mutate(comparision=(polymer == y_test)*1)

mean(corr_table$comparision)

prob_table <- left_join(data.frame(class=as.integer(y_pred),
                                   prob=y_pred_prob,
                                   y_test) ,
                        slopp_polymer_dict, by='class') %>%
  select(polymer, y_test, prob) %>%
  mutate(comparision=(polymer == y_test)*1)

mean(prob_table$comparision)
