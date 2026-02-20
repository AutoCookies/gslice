package application

import (
	"errors"
	"gslice/internal/domain"
)

const (
	CodeNotFound      = "ERR_NOT_FOUND"
	CodeExpired       = "ERR_EXPIRED"
	CodeQuotaExceeded = "ERR_QUOTA_EXCEEDED"
	CodeBadRequest    = "ERR_BAD_REQUEST"
	CodeInternal      = "ERR_INTERNAL"
)

type CodedError struct {
	Code    string
	Message string
}

func (e CodedError) Error() string { return e.Message }

func NewCodedError(code, message string) error {
	return CodedError{Code: code, Message: message}
}

func ErrorCode(err error) string {
	var ce CodedError
	if errors.As(err, &ce) {
		return ce.Code
	}
	return CodeInternal
}

func wrapDomainErr(err error) error {
	if errors.Is(err, domain.ErrSessionNotFound) {
		return NewCodedError(CodeNotFound, "session not found")
	}
	return NewCodedError(CodeInternal, "internal error")
}
