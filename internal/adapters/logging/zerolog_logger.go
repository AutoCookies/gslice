package logging

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type ZerologAdapter struct{}

func New() ZerologAdapter { return ZerologAdapter{} }

func (z ZerologAdapter) Info(msg string, kv ...any)  { z.log("info", msg, kv...) }
func (z ZerologAdapter) Error(msg string, kv ...any) { z.log("error", msg, kv...) }

func (z ZerologAdapter) log(level, msg string, kv ...any) {
	m := map[string]any{"level": level, "msg": msg, "ts": time.Now().UTC().Format(time.RFC3339Nano)}
	for i := 0; i+1 < len(kv); i += 2 {
		k, ok := kv[i].(string)
		if ok {
			m[k] = kv[i+1]
		}
	}
	b, _ := json.Marshal(m)
	fmt.Fprintln(os.Stdout, string(b))
}
