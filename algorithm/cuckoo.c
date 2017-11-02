#include "config.h"
#include "miner.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#include "sph/sph_sha2.h"

void cuckoo_regenhash(struct work *work)
{
  unsigned char hash1[32];
  sph_sha256_context ctx_sha2;
  uint32_t *cuckoo_proof = (uint32_t *)work->pow;

  sph_sha256_init(&ctx_sha2);
  sph_sha256(&ctx_sha2, cuckoo_proof, 42 * sizeof(uint32_t));
  sph_sha256_close(&ctx_sha2, hash1);
  sph_sha256(&ctx_sha2, hash1, 32);
  sph_sha256_close(&ctx_sha2, work->hash);
}
