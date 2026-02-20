package ipc

const (
	CodeNotFound           = "ERR_NOT_FOUND"
	CodeExpired            = "ERR_EXPIRED"
	CodeQuotaExceeded      = "ERR_QUOTA_EXCEEDED"
	CodeBadRequest         = "ERR_BAD_REQUEST"
	CodeInternal           = "ERR_INTERNAL"
	CodeUnsupportedVersion = "ERR_UNSUPPORTED_VERSION"
	CodeBadOp              = "ERR_BAD_OP"
)

type ErrorPayload struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}
