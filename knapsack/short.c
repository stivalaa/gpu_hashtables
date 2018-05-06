
#define DEBUG
#define STATS
/*#define KCMST*/
#define SOLNS

#define MAXITEMS 10000
#define MAXWEIGHT 10000

int weight[MAXITEMS+1];
int profit[MAXITEMS+1];

int cumweight[MAXITEMS+1];  /* cumweight[i] = sum(j = 0 .. i) weight[i]; */
int cumprofit[MAXITEMS+1];  /* cumprofit[i] = sum(j = 0.. i) profit[i];  */

int lowerbnd[MAXWEIGHT+1];  /* minimum profit for weight */
int upperbnd[MAXWEIGHT+1];  /* maximum profit for weight ignoring integrality */

int k[MAXITEMS+1][MAXWEIGHT+1];

int kcount = 0;
int kcountnr = 0;
int kcount2 = 0;
int kcount2r = 0;
int kbcount = 0;
int kbcountr = 0;

#ifdef STATS
int kbase = 0;
int klookup = 0;
int kbase2 = 0;
int klookup2 = 0;
int kbase2r = 0;
int klookup2r = 0;
int kbbase = 0;
int kbprune = 0;
int kbreval = 0;
int kblookup = 0;
int kbbaser = 0;
int kbpruner = 0;
int kbrevalr = 0;
int kblookupr = 0;
#endif

int showk(int ITEMS, int WEIGHT)
{
  int i, w;

  for (w = WEIGHT; w > 0; w--) {
    printf("%4d:",w);
    for (i = 1; i <= ITEMS; i++)
      printf("%4d ",k[i][w]);
    printf("\n");
  }
  return 1;

}

int upperbound(int i, int w) {
  int p;

  if (cumweight[i] <= w) p = cumprofit[i];
  else p = upperbnd[w];
#ifdef DEBUG
  printf("[UB = %d] ",p);
#endif
  return p;
}

int cleark(int ITEMS, int WEIGHT) {
  int i,w;

  for (i = 1; i <= ITEMS; i++)
    for (w = 0; w <= WEIGHT; w++)
      k[i][w] = 0;
  return 1;
}


/* return the maximum profit available from using the
   first i items in a knapsack of weight w 
*/
int knapsack(int i, int w) {
  int p, p2;

#ifdef DEBUG
  printf("knapsack(%d,%d)\n",i,w);
#endif 
  if (i <= 0 || w <= 0) {
#ifdef STATS
    kbase++;
#endif
    return 0;
  }
  if ((p = k[i][w]) > 0) {
#ifdef STATS
    klookup++;
#endif
    return p;
  }
  kcount++;
  p = knapsack(i-1, w);
  if (w >= weight[i]) {
    p2 = knapsack(i-1,w-weight[i]) + profit[i];
    if (p2 > p) p = p2;
  }
  k[i][w] = p;
#ifdef DEBUG
  printf("knapsack(%d,%d) = %d\n",i,w,p);
#endif 
  return p;
}

#define max(a,b)    ((a) >= (b) ? (a) : (b))
#define lookk(i,w)  ((i) >= 0 && (w) >= 0 ? k[i][w] : 0)
#define TRUE        1
#define FALSE       0

/* knapsack with no recursion */
int knapsack_nr(int i, int w) {
  int p, p2, c;
  int stackc[MAXITEMS+1];  /* call or answer */
  int stacki[MAXITEMS+1];
  int stackw[MAXITEMS+1];
  int top = 1;
  int oi = i, ow = w;
  
#ifdef DEBUG
  printf("knapsack_nr(%d,%d)\n",i,w);
#endif 
  stacki[top] = i;
  stackw[top] = w;
  stackc[top] = TRUE;
  while(top) {
    i = stacki[top];
    w = stackw[top];
    c = stackc[top];
    kcountnr++;
    top--;
#ifdef DEBUG
    printf("knapsack_nr(%d,%d,%d)\n",i,w,c);
#endif    
    if (c) {
      if (i == 1) { 
	p = (w >= weight[i] ? profit[i] : 0);
#ifdef DEBUG
        printf("knapsack_nr(%d,%d,%d) = %d\n",i,w,c,p);
#endif    	
	k[i][w] = p;
      }
      else if (lookk(i-1,w) > 0 && lookk(i-1,w - weight[i]) > 0) {
	p = max(lookk(i-1,w), 
                (w >= weight[i] ? lookk(i-1,w - weight[i]) + profit[i] : 0));
#ifdef DEBUG
        printf("knapsack_nr(%d,%d,%d) = %d\n",i,w,c,p);
#endif    	
	k[i][w] = p;
      }
      else {
	top++;
	stacki[top] = i;
	stackw[top] = w;
	stackc[top] = FALSE;
	if (i >= 1 && lookk(i-1,w) == 0) {
	  top++;
	  stacki[top] = i-1;
	  stackw[top] = w;
	  stackc[top] = TRUE;
	}
	if (i >= 1 && w >= weight[i] && lookk(i-1,w-weight[i]) == 0) {
	  top++;
	  stacki[top] = i-1;
	  stackw[top] = w - weight[i];
	  stackc[top] = TRUE;
	}
      }
    } else {
      p = max(lookk(i-1,w), 
	      (w >= weight[i] ? lookk(i-1,w - weight[i]) +profit[i] : 0));
#ifdef DEBUG
      printf("knapsack_nr(%d,%d,%d) = %d\n",i,w,c,p);
#endif    	
      k[i][w] = p;
    }
  }
  p = k[oi][ow];
#ifdef DEBUG
  printf("knapsack_nr(%d,%d) = %d\n",i,w,p);
#endif 
  return p;
}
   

/* return the maximum profit available from using the
   first i items in a knapsack of weight w, uses upperbounds
   to avoid searching second case? 
*/
int knapsack2(int i, int w) {
  int p, p2;

#ifdef DEBUG
  printf("knapsack2(%d,%d)\n",i,w);
#endif 
  if (i <= 0 || w <= 0) {
#ifdef STATS
    kbase2++;
#endif
    return 0;
  }
  if ((p = k[i][w]) > 0) {
#ifdef STATS
    klookup2++;
#endif
    return p;
  }
  kcount2++;
  p = knapsack2(i-1, w);
  if (w >= weight[i] && p < upperbound(i-1,w-weight[i]) + profit[i]) {
    p2 = knapsack2(i-1,w-weight[i]) + profit[i];
    if (p2 > p) p = p2;
  }
  k[i][w] = p;
#ifdef DEBUG
  printf("knapsack2(%d,%d) = %d\n",i,w,p);
#endif 
  return p;
}
/* return the maximum profit available from using the
   first i items in a knapsack of weight w, uses upperbounds
   to avoid searching second case? 
*/
int knapsack2r(int i, int w) {
  int p, p2;

#ifdef DEBUG
  printf("knapsack2r(%d,%d)\n",i,w);
#endif 
  if (i <= 0 || w <= 0) {
#ifdef STATS
    kbase2r++;
#endif
    return 0;
  }
  if ((p = k[i][w]) > 0) {
#ifdef STATS
    klookup2r++;
#endif
    return p;
  }
  kcount2r++;
  p = 0;
  if (w >= weight[i])
    p = knapsack2r(i-1,w-weight[i]) + profit[i];
  if (p < upperbound(i-1,w)) {
    p2 = knapsack2r(i-1, w);
    if (p2 > p) p = p2;
  }
  k[i][w] = p;
#ifdef DEBUG
  printf("knapsack2r(%d,%d) = %d\n",i,w,p);
#endif 
  return p;
}

int soln[MAXITEMS];
int solnno = 0;

int solution(int ITEMS, int WEIGHT)
{
  int i;

  solnno = 0;
  printf("SOL: ");
  sol(ITEMS,WEIGHT);
  printf("\n");
  printf("WGT: ");
  for (i = solnno-1; i >= 0; i--) 
    printf("%4d ",weight[soln[i]]);
  printf("\n");
  printf("PFT: ");
  for (i = solnno-1; i >= 0; i--) 
    printf("%4d ",profit[soln[i]]);
  printf("\n");
}

int sol(int i, int w)
{
  if (i <= 0) return;
  if (k[i][w] == k[i-1][w])
    sol(i-1,w);
  else {
    soln[solnno++] = i;
    printf("%4d ",i);
    sol(i-1,w-weight[i]);
  }
  return 1;
}

/* bounded knapsack, you have to make at least lb profit */
/* If the return value is less than the lowerbound it is an upperbound */
int knapsackb(int i, int w, int lb) {
  int p, p2;
  int ub;

#ifdef DEBUG
  printf("knapsackb(%d,%d,%d)",i,w,lb);
#endif 
  if (i <= 0 || w <= 0) {
#ifdef DEBUG
    printf("= 0 (BASE)\n");
#endif 
#ifdef STATS
    kbbase++;
#endif
    return 0;
  }
  if ((p = k[i][w]) > 0) {
#ifdef DEBUG
    printf("= %d (LOOKUP)\n",p);
#endif 
#ifdef STATS
    kblookup++;
#endif
    return p;
  }
  if (p && -p < lb) {
#ifdef DEBUG
    printf("= %d (LOOKUP PRUNE)\n",p);
#endif 
#ifdef STATS
    kbprune++;
#endif
    return -p;
  }
  if (!p && (ub = upperbound(i,w)) < lb) {
#ifdef DEBUG
    printf("= %d (NEW PRUNE)\n",-ub);
#endif 
#ifdef STATS
    kbprune++;
#endif
    k[i][w] = -ub;
    return ub;
  }
#ifdef DEBUG
  printf("\n");
#endif 
#ifdef STATS
  if (p) kbreval++;
#endif
  kbcount++;
  p = knapsackb(i-1, w, lb);
  if (p > lb) lb = p;
  if (w >= weight[i]) {
    p2 = knapsackb(i-1,w-weight[i],lb-profit[i]) + profit[i];
    if (p2 > p) p = p2;
  }
  if (p >= lb)
    k[i][w] = p;
  else
    k[i][w] = -p;
#ifdef DEBUG
  printf("knapsackb(%d,%d,%d) = %d\n",i,w,lb,p);
#endif 
  return p;
}

/* bounded knapsack, you have to make at least lb profit */
/* If the return value is less than the lowerbound it is an upperbound */
int knapsackbr(int i, int w, int lb) {
  int p, p2;
  int ub;

#ifdef DEBUG
  printf("knapsackbr(%d,%d,%d)",i,w,lb);
#endif 
  if (i <= 0 || w <= 0) {
#ifdef DEBUG
    printf("= 0 (BASE)\n");
#endif 
#ifdef STATS
    kbbaser++;
#endif
    return 0;
  }
  if ((p = k[i][w]) > 0) {
#ifdef DEBUG
    printf("= %d (LOOKUP)\n",p);
#endif 
#ifdef STATS
    kblookupr++;
#endif
    return p;
  }
  if (p && -p < lb) {
#ifdef DEBUG
    printf("= %d (LOOKUP PRUNE)\n",p);
#endif 
#ifdef STATS
    kbpruner++;
#endif
    return -p;
  }
  if (!p && (ub = upperbound(i,w)) < lb) {
#ifdef DEBUG
    printf("= %d (NEW PRUNE)\n",-ub);
#endif 
#ifdef STATS
    kbpruner++;
#endif
    k[i][w] = -ub;
    return ub;
  }
#ifdef DEBUG
  printf("\n");
#endif 
#ifdef STATS
  if (p) kbrevalr++;
#endif
  kbcountr++;
  p = 0;
  if (w >= weight[i])
    p = knapsackbr(i-1,w-weight[i],lb-profit[i]) + profit[i];
  if (p > lb) lb = p;
  p2 = knapsackbr(i-1, w, lb);
  if (p2 > p) p = p2;
  if (p >= lb)
    k[i][w] = p;
  else
    k[i][w] = -p;
#ifdef DEBUG
  printf("knapsackbr(%d,%d,%d) = %d\n",i,w,lb,p);
#endif 
  return p;
}


main() {
  int cw = 0; 
  int cp = 0;
  int i, j, w;
  
  int ITEMS;
  int WEIGHT;

  int kp, kp2, kp2r, kpb, kpbr, kpnr;
  int bi;
  float ri, rj;

#ifdef KCMST
  scanf("%d",&w); /* nodes */
  scanf("%d",&ITEMS);
  scanf("%d",&WEIGHT);
  for (i = 1; i <= ITEMS; i++) {
    scanf("%d",&w); /* node 1 */
    scanf("%d",&w); /* node 2 */
    scanf("%d",&w);
    profit[i] = w;
    scanf("%d",&w);
    weight[i] = w;
  }
#else
  scanf("%d",&ITEMS);
  scanf("%d",&WEIGHT);
  for (i = 1; i <= ITEMS; i++) {
    scanf("%d",&w);
    weight[i] = w;
    scanf("%d",&w);
    profit[i] = w;
  }
#endif



#ifdef DEBUG
  printf("WEIGHT    ");
  for (i = 1; i <= ITEMS; i++)
    printf("%3d ", weight[i]);
  printf("\n");
  printf("PROFIT    ");
  for (i = 1; i <= ITEMS; i++)
    printf("%3d ", profit[i]);
  printf("\n");
#endif

  /** selection sort them **/
  for (i = 1; i <= ITEMS; i++) {
    bi = i;
    ri = (float)profit[i]/(float)weight[i];
    for (j = i+1; j <= ITEMS; j++) {
      rj = (float)profit[j]/(float)weight[j];
      if (rj > ri) {
	ri = rj;
	bi = j;
      }
    }
    w = weight[i]; weight[i] = weight[bi]; weight[bi] = w;
    w = profit[i]; profit[i] = profit[bi]; profit[bi] = w;
  }
#ifdef DEBUG
  printf("WEIGHT    ");
  for (i = 1; i <= ITEMS; i++)
    printf("%3d ", weight[i]);
  printf("\n");
  printf("PROFIT    ");
  for (i = 1; i <= ITEMS; i++)
    printf("%3d ", profit[i]);
  printf("\n");
  printf("RATIO     ");
  for (i = 1; i <= ITEMS; i++)
    printf("%6.3f ", (float)profit[i]/(float)weight[i]);
  printf("\n");
#endif


  /** Initialize the cumulative weights */
  for (i = 1; i <= ITEMS; i++) {
    cumweight[i] = cw = weight[i] + cw;
    cumprofit[i] = cp = profit[i] + cp;
  }
#ifdef DEBUG
  printf("CUMWEIGHT ");
  for (i = 1; i <= ITEMS; i++)
    printf("%3d ", cumweight[i]);
  printf("\n");
  printf("CUMPROFIT ");
  for (i = 1; i <= ITEMS; i++)
    printf("%3d ", cumprofit[i]);
  printf("\n");
#endif
  
  /** Initialize the upperbnd profit **/
  i = 1; cp = 0; cw = 0;
  for (w = 0; w <= WEIGHT; w++) 
    if (cw + weight[i] > w) {
      lowerbnd[w] = cp;
      upperbnd[w] = cp + floor((profit[i] * (w - cw))/weight[i]);
    } else {
      lowerbnd[w] = cumprofit[i];
      upperbnd[w] = cumprofit[i];
      cw = cumweight[i];
      cp = cumprofit[i];
      i++;
    }
#ifdef DEBUG
  printf("w        ");
  for (w = 0; w <= WEIGHT; w++)
    printf("%3d ",w);
  printf("\n");
  printf("LOWERBND ");
  for (w = 0; w <= WEIGHT; w++)
    printf("%3d ", lowerbnd[w]);
  printf("\n");
  printf("UPPERBND ");
  for (w = 0; w <= WEIGHT; w++)
    printf("%3d ", upperbnd[w]);
  printf("\n");
#endif
  
 
  kp = knapsack(ITEMS,WEIGHT);
#ifdef SOLNS
  solution(ITEMS,WEIGHT);
#endif
#ifdef DEBUG
  showk(ITEMS,WEIGHT);
#endif
  cleark(ITEMS,WEIGHT);
  kpnr = knapsack_nr(ITEMS,WEIGHT);
#ifdef SOLNS
  solution(ITEMS,WEIGHT);
#endif
#ifdef DEBUG
  showk(ITEMS,WEIGHT);
#endif
  /*
  cleark(ITEMS,WEIGHT);
  kp2 = knapsack2(ITEMS,WEIGHT);
#ifdef SOLNS
  solution(ITEMS,WEIGHT);
#endif
#ifdef DEBUG
  showk(ITEMS,WEIGHT);
#endif
  cleark(ITEMS,WEIGHT);
  kp2r = knapsack2r(ITEMS,WEIGHT);
#ifdef SOLNS
  solution(ITEMS,WEIGHT);
#endif
#ifdef DEBUG
  showk(ITEMS,WEIGHT);
#endif
  cleark(ITEMS,WEIGHT);
  kpb = knapsackb(ITEMS,WEIGHT,lowerbnd[WEIGHT]);
#ifdef SOLNS
  solution(ITEMS,WEIGHT);
#endif
#ifdef DEBUG
  showk(ITEMS,WEIGHT);
#endif
  cleark(ITEMS,WEIGHT);
  kpbr = knapsackbr(ITEMS,WEIGHT,lowerbnd[WEIGHT]);
#ifdef SOLNS
  solution(ITEMS,WEIGHT);
#endif
#ifdef DEBUG
  showk(ITEMS,WEIGHT);
#endif
  */
  printf("knapsack  = %d, count = %d\n", kp, kcount);
  printf("knapsack_nr = %d, count = %d\n", kpnr, kcountnr);
  /*
  printf("knapsack2 = %d, count = %d\n", kp2, kcount2);
  printf("knapsack2r= %d, count = %d\n", kp2r, kcount2r);
  printf("knapsackb = %d, count = %d\n", kpb, kbcount);
  printf("knapsackbr= %d, count = %d\n", kpbr, kbcountr);
#ifdef STATS
  printf("KNAPSACK   base=%d, lookup=%d\n",kbase,klookup);
  printf("KNAPSACK2  base=%d, lookup=%d\n",kbase2,klookup2);
  printf("KNAPSACK2R base=%d, lookup=%d\n",kbase2r,klookup2r);
  printf("KNAPSACKB  base=%d, lookup=%d, reeval=%d, prune=%d\n",kbbase,kblookup, kbreval, kbprune);
  printf("KNAPSACKBR base=%d, lookup=%d, reeval=%d, prune=%d\n",kbbaser,kblookupr, kbrevalr, kbpruner);
#endif
  */
}

