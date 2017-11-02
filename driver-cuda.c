/*
 * Copyright 2011-2012 Con Kolivas
 * Copyright 2011-2012 Luke Dashjr
 * Copyright 2010 Jeff Garzik
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "config.h"

#ifdef HAVE_CURSES
#include <curses.h>
#endif

#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <signal.h>
#include <sys/types.h>
#include <limits.h>

#ifndef WIN32
#include <sys/resource.h>
#endif
#include <ccan/opt/opt.h>

#include "compat.h"
#include "miner.h"
#include "config_parser.h"
#include "driver-cuda.h"
#include "findnonce.h"
#include "util.h"

static void cuda_detect(void)
{
  int i;

  // BUGBUG cudaGetDeviceCount
  nDevs = 1;
  if (nDevs < 0) {
    applog(LOG_ERR, "cudaGetDeviceCount returned error, no GPUs usable");
    nDevs = 0;
  }

  if (!nDevs)
    return;

  cuda_drv.max_diff = 65536;

  for (i = 0; i < nDevs; ++i) {
    struct cgpu_info *cgpu;

    cgpu = &gpus[i];
    cgpu->deven = DEV_ENABLED;
    cgpu->drv = &cuda_drv;
    cgpu->thr = NULL;
    cgpu->device_id = i;
    if (cgpu->threads < 1)
      cgpu->threads = 1;
    cgpu->virtual_gpu = i;
    cgpu->algorithm = default_profile.algorithm;
    add_cgpu(cgpu);
  }
}

struct cuda_thread_data {
    uint32_t cuckoo_pow[42]; /* must be first */
    uint32_t res[MAXBUFFERS];
};

int cuckoo_scanhash(const char *header, uint32_t nonce, uint32_t *proof);

static int64_t cuda_scanhash(struct thr_info *thr, struct work *work,
  int64_t __maybe_unused max_nonce)
{
	uint32_t max_nonce_batch = 50, nonce;
    struct cuda_thread_data *thrdata = thr->cgpu_data;
    struct cgpu_info *gpu = thr->cgpu;
    int found = gpu->algorithm.found_idx;
    uint32_t nonce_offset = work->blk.nonce;
    int rc;

	if (UINT_MAX - nonce_offset < max_nonce_batch) {
		work->blk.nonce = 0;
		max_nonce_batch = UINT_MAX - nonce_offset;
	} else {
		work->blk.nonce += max_nonce_batch;
	}
	

    thrdata->res[found] = 0;
	for (nonce = nonce_offset; nonce < (nonce_offset + max_nonce_batch); nonce++) {
        rc = cuckoo_scanhash(work->data, nonce, work->pow);
        if (rc == 0) {
            thrdata->res[found]++;
            thrdata->res[0] = nonce;
            applog(LOG_DEBUG, "[THR%d] Found!", thr->id);
            postcalc_hash_async(thr, work, thrdata->res);
            break;
        }
    }

    return nonce;
}

static bool cuda_thread_init(struct thr_info *thr)
{
  const int thr_id = thr->id;
  struct cgpu_info *gpu = thr->cgpu;
  struct cuda_thread_data *thrdata;
  thrdata = (struct cuda_thread_data *)calloc(1, sizeof(*thrdata));
  thr->cgpu_data = thrdata;

  if (!thrdata) {
    applog(LOG_ERR, "Failed to calloc in opencl_thread_init");
    return false;
  }
  return true;
}

static void cuda_thread_shutdown(struct thr_info *thr)
{
  free(thr->cgpu_data);
  thr->cgpu_data = NULL;
}

struct device_drv cuda_drv = {
  /*.drv_id = */      DRIVER_cuda,
  /*.dname = */     "cuda",
  /*.name = */      "CPU",
  /*.drv_detect = */ cuda_detect,
  /*.reinit_device = */   NULL, //reinit_cuda_device,
  /*.get_statline_before = */ NULL,
  /*.get_statline = */    NULL, //get_cuda_statline,
  /*.api_data = */    NULL,
  /*.get_stats = */   NULL,
  /*.identify_device = */   NULL,
  /*.set_device = */    NULL,
  /*.thread_prepare = */    NULL, //cuda_thread_prepare,
  /*.can_limit_work = */    NULL,
  /*.thread_init = */   cuda_thread_init,
  /*.prepare_work = */    NULL, //cuda_prepare_work,
  /*.hash_work = */   NULL,
  /*.scanhash = */   cuda_scanhash,
  /*.scanwork = */    NULL,
  /*.queue_full = */    NULL,
  /*.flush_work = */    NULL,
  /*.update_work = */   NULL,
  /*.hw_error = */    NULL,
  /*.thread_shutdown = */  cuda_thread_shutdown,
  /*.thread_enable =*/    NULL,
  false,
  0,
  0
};
