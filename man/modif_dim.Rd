% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/materialmodifier.R
\name{modif_dim}
\alias{modif_dim}
\title{Check the scale information of an image}
\usage{
modif_dim(im)
}
\arguments{
\item{im}{An image.}
}
\value{
A list of depth (number of scale subband images), indexes of high amplitude subbands,
and indexes of low amplitude subbands.
}
\description{
Check the scale information of an image
}
\examples{
modif_dim(face)
}
