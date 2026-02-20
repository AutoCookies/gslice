package domain

func Reserve(used, limit, req uint64) (uint64, error) {
	if req > limit || used > limit || req > limit-used {
		return used, ErrQuotaExceeded
	}
	return used + req, nil
}

func Release(used, size uint64) uint64 {
	if size > used {
		return 0
	}
	return used - size
}
