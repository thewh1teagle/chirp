package main

import "os"

var version = "dev"
var commit = "dev"

func main() {
	if err := newRootCommand().Execute(); err != nil {
		os.Exit(1)
	}
}
