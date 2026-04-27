package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/thewh1teagle/chirp/chirp-runner/internal/chirpc"
	"github.com/thewh1teagle/chirp/chirp-runner/internal/server"
)

func newRootCommand() *cobra.Command {
	rootCmd := &cobra.Command{
		Use:     "chirp",
		Short:   "Qwen3-TTS runner",
		Version: version,
	}
	rootCmd.AddCommand(newServeCommand(), newSpeakCommand(), newLanguagesCommand())
	return rootCmd
}

func newServeCommand() *cobra.Command {
	var host, modelPath, codecPath string
	var port, maxTokens, topK int
	var temperature float32

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start a Qwen3-TTS HTTP runner",
		RunE: func(cmd *cobra.Command, args []string) error {
			s := server.New()
			s.Version = version
			s.Commit = commit
			if modelPath != "" || codecPath != "" {
				if modelPath == "" || codecPath == "" {
					return fmt.Errorf("--model and --codec must be provided together")
				}
				if err := s.LoadModel(chirpc.Params{
					ModelPath:   modelPath,
					CodecPath:   codecPath,
					MaxTokens:   maxTokens,
					Temperature: temperature,
					TopK:        topK,
				}); err != nil {
					return fmt.Errorf("error loading model: %w", err)
				}
			}
			return server.ListenAndServe(host, port, s)
		},
	}
	cmd.Flags().StringVar(&host, "host", "127.0.0.1", "host to bind to")
	cmd.Flags().IntVarP(&port, "port", "p", 0, "port to listen on (0 = auto-assign)")
	cmd.Flags().StringVar(&modelPath, "model", "", "Qwen3-TTS model GGUF")
	cmd.Flags().StringVar(&codecPath, "codec", "", "Qwen3-TTS codec GGUF")
	cmd.Flags().IntVar(&maxTokens, "max-tokens", 0, "maximum generated frames (0 = runtime default)")
	cmd.Flags().Float32Var(&temperature, "temperature", 0.9, "sampling temperature")
	cmd.Flags().IntVar(&topK, "top-k", 50, "top-k sampling")
	return cmd
}

func newSpeakCommand() *cobra.Command {
	var modelPath, codecPath, text, refPath, outputPath string
	var language string
	var maxTokens, topK int
	var temperature float32

	cmd := &cobra.Command{
		Use:   "speak",
		Short: "Generate a WAV file",
		RunE: func(cmd *cobra.Command, args []string) error {
			if modelPath == "" || codecPath == "" || text == "" || outputPath == "" {
				return fmt.Errorf("--model, --codec, --text, and --output are required")
			}
			ctx, err := chirpc.New(chirpc.Params{
				ModelPath:   modelPath,
				CodecPath:   codecPath,
				MaxTokens:   maxTokens,
				Temperature: temperature,
				TopK:        topK,
			})
			if err != nil {
				return err
			}
			defer ctx.Close()
			if err := ctx.SynthesizeToFile(text, refPath, outputPath, language); err != nil {
				return err
			}
			enc := json.NewEncoder(os.Stdout)
			return enc.Encode(map[string]string{"output": outputPath})
		},
	}
	cmd.Flags().StringVar(&modelPath, "model", "", "Qwen3-TTS model GGUF")
	cmd.Flags().StringVar(&codecPath, "codec", "", "Qwen3-TTS codec GGUF")
	cmd.Flags().StringVar(&text, "text", "", "text to synthesize")
	cmd.Flags().StringVar(&refPath, "ref", "", "optional voice reference WAV")
	cmd.Flags().StringVarP(&outputPath, "output", "o", "", "output WAV path")
	cmd.Flags().StringVar(&language, "language", "auto", "target language")
	cmd.Flags().IntVar(&maxTokens, "max-tokens", 0, "maximum generated frames (0 = runtime default)")
	cmd.Flags().Float32Var(&temperature, "temperature", 0.9, "sampling temperature")
	cmd.Flags().IntVar(&topK, "top-k", 50, "top-k sampling")
	return cmd
}

func newLanguagesCommand() *cobra.Command {
	var modelPath, codecPath string
	cmd := &cobra.Command{
		Use:   "languages",
		Short: "List supported synthesis languages",
		RunE: func(cmd *cobra.Command, args []string) error {
			if modelPath == "" || codecPath == "" {
				return fmt.Errorf("--model and --codec are required")
			}
			ctx, err := chirpc.New(chirpc.Params{
				ModelPath: modelPath,
				CodecPath: codecPath,
			})
			if err != nil {
				return err
			}
			defer ctx.Close()
			names := []string{"auto"}
			for _, language := range ctx.Languages() {
				names = append(names, language.Name)
			}
			enc := json.NewEncoder(os.Stdout)
			return enc.Encode(map[string]any{
				"languages": names,
				"items":     ctx.Languages(),
			})
		},
	}
	cmd.Flags().StringVar(&modelPath, "model", "", "Qwen3-TTS model GGUF")
	cmd.Flags().StringVar(&codecPath, "codec", "", "Qwen3-TTS codec GGUF")
	return cmd
}
