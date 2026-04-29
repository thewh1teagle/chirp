package kokoroc

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/thewh1teagle/chirp/chirp-runner/internal/chirpc"
)

type Params struct {
	ModelPath      string
	VoicesPath     string
	EspeakDataPath string
	Voice          string
	Language       string
	Speed          float32
}

type Context struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	reader *bufio.Reader
	mu     sync.Mutex
	nextID int
	err    string
}

type response struct {
	ID        int      `json:"id"`
	OK        bool     `json:"ok"`
	Error     string   `json:"error,omitempty"`
	Languages []string `json:"languages,omitempty"`
}

func New(params Params) (*Context, error) {
	if params.ModelPath == "" {
		return nil, errors.New("model path is required")
	}
	if params.VoicesPath == "" {
		return nil, errors.New("voices path is required")
	}

	worker, err := findWorker()
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(worker)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	ctx := &Context{
		cmd:    cmd,
		stdin:  stdin,
		reader: bufio.NewReader(stdout),
		nextID: 1,
	}
	if err := ctx.call(map[string]any{
		"method":           "load",
		"model_path":       params.ModelPath,
		"voices_path":      params.VoicesPath,
		"espeak_data_path": params.EspeakDataPath,
		"voice":            defaultString(params.Voice, "af_heart"),
		"language":         kokoroLanguage(params.Language),
		"speed":            defaultFloat(params.Speed, 1.0),
	}, nil); err != nil {
		ctx.Close()
		return nil, err
	}
	return ctx, nil
}

func (c *Context) Close() {
	if c == nil {
		return
	}
	_ = c.call(map[string]any{"method": "shutdown"}, nil)
	if c.stdin != nil {
		_ = c.stdin.Close()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		_ = c.cmd.Process.Kill()
		_, _ = c.cmd.Process.Wait()
	}
}

func (c *Context) Error() string {
	if c == nil {
		return "kokoro worker context is nil"
	}
	return c.err
}

func (c *Context) Languages() []chirpc.Language {
	var resp response
	if err := c.call(map[string]any{"method": "languages"}, &resp); err != nil {
		c.err = err.Error()
		return kokoroLanguages([]string{"en-us", "en", "es", "fr", "ja", "hi", "it", "pt-br"})
	}
	return kokoroLanguages(resp.Languages)
}

func (c *Context) SynthesizeToFile(text, voice string, outputPath, language string) error {
	return c.call(map[string]any{
		"method":      "synthesize",
		"input":       text,
		"output_path": outputPath,
		"language":    kokoroLanguage(language),
		"voice":       voice,
	}, nil)
}

func kokoroLanguages(languages []string) []chirpc.Language {
	out := make([]chirpc.Language, 0, len(languages))
	for i, language := range languages {
		if language == "auto" {
			continue
		}
		out = append(out, chirpc.Language{Name: language, ID: i})
	}
	return out
}

func kokoroLanguage(language string) string {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "", "auto", "english", "american", "en-us":
		return "en-us"
	case "british", "en", "en-gb":
		return "en"
	case "spanish", "es":
		return "es"
	case "french", "fr", "fr-fr":
		return "fr"
	case "japanese", "ja":
		return "ja"
	case "hindi", "hi":
		return "hi"
	case "italian", "it":
		return "it"
	case "portuguese", "pt", "pt-br":
		return "pt-br"
	default:
		return strings.TrimSpace(language)
	}
}

func (c *Context) call(payload map[string]any, out *response) error {
	if c == nil || c.stdin == nil || c.reader == nil {
		return errors.New("kokoro worker context is nil")
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	id := c.nextID
	c.nextID++
	payload["id"] = id
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := c.stdin.Write(append(data, '\n')); err != nil {
		return err
	}
	line, err := c.reader.ReadBytes('\n')
	if err != nil {
		return err
	}
	var resp response
	if err := json.Unmarshal(line, &resp); err != nil {
		return fmt.Errorf("invalid kokoro worker response: %w: %s", err, strings.TrimSpace(string(line)))
	}
	if resp.ID != id {
		return fmt.Errorf("unexpected kokoro worker response id %d, expected %d", resp.ID, id)
	}
	if !resp.OK {
		if resp.Error == "" {
			resp.Error = "kokoro worker request failed"
		}
		c.err = resp.Error
		return errors.New(resp.Error)
	}
	c.err = ""
	if out != nil {
		*out = resp
	}
	return nil
}

func findWorker() (string, error) {
	name := "chirp-kokoro-worker"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	candidates := []string{}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		candidates = append(candidates, filepath.Join(dir, name), filepath.Join(dir, "binaries", name))
		if matches, err := filepath.Glob(filepath.Join(dir, targetSidecarPattern())); err == nil {
			candidates = append(candidates, matches...)
		}
		if matches, err := filepath.Glob(filepath.Join(dir, "binaries", targetSidecarPattern())); err == nil {
			candidates = append(candidates, matches...)
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates,
			filepath.Join(cwd, name),
			filepath.Join(cwd, "runtimes", "kokoro", "build-worker", name),
			filepath.Join(cwd, "..", "runtimes", "kokoro", "build-voices", name),
			filepath.Join(cwd, "..", "runtimes", "kokoro", "build-worker", name),
			filepath.Join(cwd, "..", "runtimes", "kokoro", "build", name),
		)
	}
	candidates = append(candidates,
		filepath.Join("..", "runtimes", "kokoro", "build-worker", name),
		filepath.Join("..", "runtimes", "kokoro", "build-voices", name),
		filepath.Join("runtimes", "kokoro", "build-worker", name),
		filepath.Join("runtimes", "kokoro", "build-voices", name),
	)
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}
	if path, err := exec.LookPath(name); err == nil {
		return path, nil
	}
	return "", fmt.Errorf("kokoro worker not found: %s", name)
}

func targetSidecarPattern() string {
	if runtime.GOOS == "windows" {
		return "chirp-kokoro-worker-*.exe"
	}
	return "chirp-kokoro-worker-*"
}

func defaultString(value, fallback string) string {
	if value == "" || value == "auto" {
		return fallback
	}
	return value
}

func defaultFloat(value, fallback float32) float32 {
	if value <= 0 {
		return fallback
	}
	return value
}
