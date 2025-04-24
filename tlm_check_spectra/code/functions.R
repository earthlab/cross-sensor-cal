# Utility ----


#' Ensure Directories Exist
#'
#' This function checks if one or more directories exist at the specified paths,
#' and creates any that do not exist.
#'
#' @param path A character string or a vector of strings specifying directory paths.
#' @return A character vector of all directory paths that were checked/created.
#' @examples
#' # Ensure a single directory
#' dir_ensure("data")
#'
#' # Ensure multiple directories
#' dir_ensure(c("data", "output", "logs"))
#'
#' @export
dir_ensure <- function(path) {
  if (!is.character(path)) {
    stop("`path` must be a character string or a vector of character strings.")
  }
  
  created_paths <- character()
  
  for (p in path) {
    if (!dir.exists(p)) {
      tryCatch({
        dir.create(p, recursive = TRUE)
        message("Directory created: ", p)
        created_paths <- c(created_paths, p)
      }, error = function(e) {
        warning("Failed to create directory: ", p, " — ", conditionMessage(e))
      })
    } else {
      message("Directory already exists: ", p)
    }
  }
  
  return(invisible(path))
}


install_and_load_packages <- function(package_list, auto_install = "n") {
  # Ensure pak is available
  if (!requireNamespace("pak", quietly = TRUE)) {
    cat("The 'pak' package is required for fast installation of packages, installing now.\n")
    install.packages("pak")
  }
  
  # Helper: Extract base name of a package for require()
  parse_pkg_name <- function(pkg) {
    if (grepl("/", pkg)) {
      sub("^.+/(.+?)(@.+)?$", "\\1", pkg)  # GitHub: extract repo name
    } else {
      sub("@.*$", "", pkg)  # CRAN: remove @version if present
    }
  }
  
  # Classify and separate packages
  missing_pkgs <- c()
  for (pkg in package_list) {
    pkg_name <- parse_pkg_name(pkg)
    if (!requireNamespace(pkg_name, quietly = TRUE)) {
      missing_pkgs <- c(missing_pkgs, pkg)
    }
  }
  
  # Install missing ones (CRAN or GitHub), with version support
  if (length(missing_pkgs) > 0) {
    pak::pkg_install(missing_pkgs, upgrade = TRUE)
  }
  
  # Load all packages
  for (pkg in package_list) {
    pkg_name <- parse_pkg_name(pkg)
    success <- require(pkg_name, character.only = TRUE, quietly = TRUE)
    if (!success) cat("Failed to load package:", pkg_name, "\n")
  }
  
  cat("All specified packages installed and loaded.\n")
}