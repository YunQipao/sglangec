/* Minimal one-shot RC QP RDMA Write. TCP carries QP/MR metadata and completion ack. */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200112L
#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdint.h>
#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MAGIC 0x4b564543u
#define MAX_BYTES (256u * 1024u * 1024u)
struct wire { uint32_t magic, qpn, psn, lid, rkey; uint64_t addr, length; uint8_t gid[16]; };

static void die(const char *s) { perror(s); exit(2); }
static void full_read(int fd, void *p, size_t n) {
  char *b=p; while(n) { ssize_t r=read(fd,b,n); if(r<=0) die("read"); b+=r; n-=r; }
}
static void full_write(int fd, const void *p, size_t n) {
  const char *b=p; while(n) { ssize_t r=write(fd,b,n); if(r<=0) die("write"); b+=r; n-=r; }
}
static double now_s(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
static int tcp_connect(const char *host, const char *port) {
  struct addrinfo hints={0},*res,*it; hints.ai_socktype=SOCK_STREAM; hints.ai_family=AF_UNSPEC;
  if(getaddrinfo(host,port,&hints,&res)) { fprintf(stderr,"resolve failed\n"); exit(2); }
  int fd=-1; for(int attempt=0;attempt<300;attempt++){ for(it=res;it;it=it->ai_next){ fd=socket(it->ai_family,it->ai_socktype,it->ai_protocol); if(fd>=0&&connect(fd,it->ai_addr,it->ai_addrlen)==0)break; if(fd>=0)close(fd); fd=-1; } if(fd>=0)break; usleep(100000); }
  freeaddrinfo(res); if(fd<0)die("connect"); return fd;
}
static int tcp_listen(const char *host, const char *port) {
  struct addrinfo h={0},*r,*i; h.ai_socktype=SOCK_STREAM; h.ai_family=AF_UNSPEC;
  if(!host || !*host) h.ai_flags=AI_PASSIVE;
  if(getaddrinfo(host,port,&h,&r)) { fprintf(stderr,"bind resolve failed\n"); exit(2); }
  int s=-1; for(i=r;i;i=i->ai_next){ s=socket(i->ai_family,i->ai_socktype,i->ai_protocol); if(s<0)continue; int one=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one)); if(bind(s,i->ai_addr,i->ai_addrlen)==0&&listen(s,1)==0)break; close(s);s=-1; }
  freeaddrinfo(r); if(s<0)die("listen"); int c=accept(s,NULL,NULL); if(c<0)die("accept"); close(s); return c;
}
static struct ibv_context *open_device(const char *name) {
  int n=0; struct ibv_device **list=ibv_get_device_list(&n); if(!list)die("ibv_get_device_list");
  struct ibv_context *ctx=NULL; for(int i=0;i<n;i++) if(!name||!strcmp(name,ibv_get_device_name(list[i]))) { ctx=ibv_open_device(list[i]); break; }
  ibv_free_device_list(list); if(!ctx){fprintf(stderr,"no requested RDMA device\n");exit(2);} return ctx;
}
static struct ibv_qp *make_qp(struct ibv_pd *pd, struct ibv_cq *cq) {
  struct ibv_qp_init_attr a={0}; a.send_cq=cq;a.recv_cq=cq;a.qp_type=IBV_QPT_RC;a.cap.max_send_wr=4;a.cap.max_recv_wr=4;a.cap.max_send_sge=1;a.cap.max_recv_sge=1;
  struct ibv_qp *qp=ibv_create_qp(pd,&a); if(!qp)die("ibv_create_qp"); return qp;
}
static void init_qp(struct ibv_qp *qp, uint8_t port_num) {
  struct ibv_qp_attr a={0}; a.qp_state=IBV_QPS_INIT;a.pkey_index=0;a.port_num=port_num;a.qp_access_flags=IBV_ACCESS_REMOTE_WRITE;
  if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))die("QP INIT");
}
static void exchange(struct ibv_context *ctx, struct ibv_qp *qp, int fd, struct wire *local, struct wire *remote, int server, uint8_t port_num, int gid_index) {
  struct ibv_port_attr pa; if(ibv_query_port(ctx,port_num,&pa))die("ibv_query_port"); local->lid=pa.lid;
  if(gid_index>=0){union ibv_gid gid;if(ibv_query_gid(ctx,port_num,gid_index,&gid))die("ibv_query_gid");memcpy(local->gid,&gid,16);}
  struct ibv_qp_attr a={0};
  local->qpn=qp->qp_num; local->psn=(uint32_t)rand()&0xffffff;
  if(server){ full_read(fd,remote,sizeof(*remote)); full_write(fd,local,sizeof(*local)); }
  else { full_write(fd,local,sizeof(*local)); full_read(fd,remote,sizeof(*remote)); }
  if(remote->magic!=MAGIC||remote->length!=local->length) {fprintf(stderr,"peer metadata mismatch\n");exit(2);}
  memset(&a,0,sizeof(a)); a.qp_state=IBV_QPS_RTR;a.path_mtu=IBV_MTU_1024;a.dest_qp_num=remote->qpn;a.rq_psn=remote->psn;a.max_dest_rd_atomic=1;a.min_rnr_timer=12;
  a.ah_attr.is_global=(gid_index>=0);a.ah_attr.dlid=remote->lid;a.ah_attr.sl=0;a.ah_attr.src_path_bits=0;a.ah_attr.port_num=port_num;
  if(gid_index>=0){memcpy(&a.ah_attr.grh.dgid,remote->gid,16);a.ah_attr.grh.sgid_index=(uint8_t)gid_index;a.ah_attr.grh.hop_limit=1;}
  if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER))die("QP RTR");
  memset(&a,0,sizeof(a));a.qp_state=IBV_QPS_RTS;a.timeout=14;a.retry_cnt=7;a.rnr_retry=7;a.sq_psn=local->psn;a.max_rd_atomic=1;
  if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC))die("QP RTS");
}
static void post_receive(struct ibv_qp *qp, void *buf, size_t len, uint32_t lkey) {
  struct ibv_sge sge={.addr=(uintptr_t)buf,.length=(uint32_t)len,.lkey=lkey};
  struct ibv_recv_wr wr={.wr_id=2,.sg_list=&sge,.num_sge=1},*bad=NULL;
  if(ibv_post_recv(qp,&wr,&bad))die("ibv_post_recv");
}
static struct ibv_wc wait_completion(struct ibv_cq *cq) {
  struct ibv_wc wc;
  for(int i=0;i<30000;i++){
    int n=ibv_poll_cq(cq,1,&wc);
    if(n<0){fprintf(stderr,"ibv_poll_cq failed\n");exit(3);}
    if(n==1)return wc;
    usleep(1000);
  }
  fprintf(stderr,"RDMA completion timeout\n");exit(3);
}
static void cleanup(struct ibv_qp *qp, struct ibv_mr *mr, struct ibv_cq *cq, struct ibv_pd *pd, struct ibv_context *ctx, void *buf) {
  if(qp) ibv_destroy_qp(qp);
  if(mr) ibv_dereg_mr(mr);
  if(cq) ibv_destroy_cq(cq);
  if(pd) ibv_dealloc_pd(pd);
  if(ctx) ibv_close_device(ctx);
  free(buf);
}
int main(int argc,char **argv) {
  if(argc<2){fprintf(stderr,"usage: %s send HOST PORT DEVICE FILE | recv PORT DEVICE FILE\n",argv[0]);return 2;}
  int send=!strcmp(argv[1],"send"),fd=-1;
  if(strcmp(argv[1],"send") && strcmp(argv[1],"recv")){fprintf(stderr,"invalid mode\n");return 2;}
  if(argc!=(send?6:5)){fprintf(stderr,"invalid arguments\n");return 2;}
  const char *host=send?argv[2]:NULL,*port=send?argv[3]:argv[2],*dev=send?argv[4]:argv[3],*file=send?argv[5]:argv[4];
  double total_started=now_s();
  FILE *f=fopen(file,send?"rb":"wb");if(!f)die("open payload");
  if(send){if(fseek(f,0,SEEK_END))die("seek");long n=ftell(f);if(n<=0||n>MAX_BYTES) {fprintf(stderr,"invalid payload size\n");return 2;} rewind(f);}
  size_t len; if(send){fseek(f,0,SEEK_END);len=(size_t)ftell(f);rewind(f);}else{uint64_t n;fd=tcp_listen(getenv("KV_EC_RDMA_BIND_HOST"),port);full_read(fd,&n,sizeof(n));len=(size_t)be64toh(n);if(!len||len>MAX_BYTES){fprintf(stderr,"invalid payload size\n");return 2;}}
  void *buf=NULL;if(posix_memalign(&buf,4096,len))die("aligned alloc");if(send&&fread(buf,1,len,f)!=len)die("read payload");
  double device_started=now_s();
  struct ibv_context *ctx=open_device(strcmp(dev,"-")?dev:NULL);struct ibv_pd *pd=ibv_alloc_pd(ctx);if(!pd)die("ibv_alloc_pd");struct ibv_cq *cq=ibv_create_cq(ctx,16,NULL,NULL,0);if(!cq)die("ibv_create_cq");
  double mr_started=now_s();
  int access=IBV_ACCESS_LOCAL_WRITE|(send?0:IBV_ACCESS_REMOTE_WRITE);struct ibv_mr *mr=ibv_reg_mr(pd,buf,len,access);if(!mr)die("ibv_reg_mr");
  double mr_ms=(now_s()-mr_started)*1000.0;
  double qp_started=now_s();
  struct ibv_qp *qp=make_qp(pd,cq);
  uint8_t port_num=(uint8_t)atoi(getenv("KV_EC_RDMA_PORT")?getenv("KV_EC_RDMA_PORT"):"1");
  int gid_index=atoi(getenv("KV_EC_RDMA_GID_INDEX")?getenv("KV_EC_RDMA_GID_INDEX"):"-1");
  init_qp(qp,port_num);
  if(send)fd=tcp_connect(host,port);
  if(send){uint64_t n=htobe64(len);full_write(fd,&n,sizeof(n));}
  struct wire local={.magic=MAGIC,.rkey=mr->rkey,.addr=(uintptr_t)buf,.length=len},remote={0};
  if(!send)post_receive(qp,buf,len,mr->lkey);
  double qp_ms=(now_s()-qp_started)*1000.0;
  double exchange_started=now_s();
  exchange(ctx,qp,fd,&local,&remote,!send,port_num,gid_index);
  double exchange_ms=(now_s()-exchange_started)*1000.0;
  double completion_started=now_s();
  if(send){struct ibv_sge sg={.addr=(uintptr_t)buf,.length=(uint32_t)len,.lkey=mr->lkey};struct ibv_send_wr wr={0},*bad=NULL;wr.wr_id=1;wr.sg_list=&sg;wr.num_sge=1;wr.opcode=IBV_WR_RDMA_WRITE_WITH_IMM;wr.imm_data=htonl((uint32_t)len);wr.send_flags=IBV_SEND_SIGNALED;wr.wr.rdma.remote_addr=remote.addr;wr.wr.rdma.rkey=remote.rkey;if(ibv_post_send(qp,&wr,&bad))die("ibv_post_send");struct ibv_wc wc=wait_completion(cq);if(wc.status!=IBV_WC_SUCCESS){fprintf(stderr,"RDMA send completion failed: %s\n",ibv_wc_status_str(wc.status));exit(3);}uint8_t ack;full_read(fd,&ack,1);if(ack!=1){fprintf(stderr,"receiver rejected payload\n");exit(3);} }
  else { struct ibv_wc wc=wait_completion(cq);if(wc.status!=IBV_WC_SUCCESS||wc.opcode!=IBV_WC_RECV_RDMA_WITH_IMM||ntohl(wc.imm_data)!=(uint32_t)len){fprintf(stderr,"RDMA receive completion failed: %s\n",ibv_wc_status_str(wc.status));exit(3);}if(fwrite(buf,1,len,f)!=len)die("write payload");uint8_t ack=1;full_write(fd,&ack,1); }
  double completion_ms=(now_s()-completion_started)*1000.0;
  /* This line is deliberately machine readable.  The TCP control channel
   * carries only QP/MR metadata and the acknowledgement; ``bytes`` above was
   * written by an RDMA work request and completed by the CQ. */
  printf("KV_EC_RDMA_STATS bytes=%zu device_setup_ms=%.3f memory_registration_ms=%.3f qp_setup_ms=%.3f exchange_ms=%.3f completion_ms=%.3f total_ms=%.3f qp_num=%u remote_qpn=%u gid_index=%d port_num=%u completion_status=SUCCESS\n", len, (mr_started-device_started)*1000.0, mr_ms, qp_ms, exchange_ms, completion_ms, (now_s()-total_started)*1000.0, qp->qp_num, remote.qpn, gid_index, port_num);
  fflush(stdout);
  fclose(f);close(fd);cleanup(qp,mr,cq,pd,ctx,buf);return 0;
}
