library(tidyverse)
load("forstmann.rda")

hist(forstmann$rt)
mean(forstmann$rt)
sd(forstmann$rt)


summary <- forstmann %>%
  mutate(response=S==R) %>%
  group_by(subjects, E, response) %>%
  summarise(
    mean=mean(rt),
    sd=sd(rt),
    min=min(rt),
    max=max(rt),
    n=n()
  ) %>%
  ungroup() |> print(n=100)

ggplot(summary, aes(x=sd, y=mean)) + geom_point()

pad_vector <- function(vec, n) {
  length(vec) <- n
  vec[is.na(vec)] <- NA
  return(vec)
}

pad_data <- function(data, n) {
  df <- lapply(data, pad_vector, n=n)
  df <- as.data.frame(df)
  colnames(df) <- colnames(data)
  return(df)
}

df <- forstmann %>%
  group_by(subjects, E) %>%
  mutate(response=as.integer(S==R), observed=1) %>%
  nest() %>%
  mutate(
    data = lapply(data, pad_data, n= 350)
  ) %>%
  unnest(cols=c(data)) %>%
  ungroup() %>%
  select(subjects, E, rt, response, observed)
df[is.na(df)] <- 0
colnames(df) <- c("subject", "condition", "rt", "response", "observed")


write.csv(df, file="data/forstmann.csv", row.names=FALSE)
