package ports

type Logger interface {
	Info(msg string, kv ...any)
	Error(msg string, kv ...any)
}
