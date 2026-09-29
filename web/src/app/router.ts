import { createRouter, createWebHistory, type RouteRecordRaw, type Router } from 'vue-router'
import ScanPage from '@/app/scan/ScanPage.vue'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: { name: 'scan' } },
  { path: '/scan', name: 'scan', component: ScanPage },
  { path: '/wines/:slug', name: 'wine', component: () => import('@/app/wine/WinePage.vue') },
  { path: '/scan/:scanId/similar', name: 'similar', component: () => import('@/app/similar/SimilarPage.vue') },
  { path: '/:pathMatch(.*)*', redirect: { name: 'scan' } }
]

export const createAppRouter = (): Router => {
  return createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
    scrollBehavior: (_to, _from, savedPosition) => savedPosition ?? { top: 0 }
  })
}
