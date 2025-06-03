import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "机器学习教程",
  description: "从零开始的机器学习实践教程",
  lang: 'zh-CN',
  base: '/simple-ml-code/',
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '教程', link: '/chapters/chapter1' }
    ],
    sidebar: [
      {
        text: '目录',
        items: [
          { text: '第1章：线性回归', link: '/chapters/chapter1' },
          { text: '第2章：多项式回归', link: '/chapters/chapter2' },
          { text: '第3章：决策树', link: '/chapters/chapter3' },
          { text: '第4章：支持向量机', link: '/chapters/chapter4' },
          { text: '第5章：K-means聚类', link: '/chapters/chapter5' },
          { text: '第6章：朴素贝叶斯', link: '/chapters/chapter6' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/ACGpp/simple-ml-code' }
    ],
    footer: {
      message: '用心打造的机器学习教程',
      copyright: 'Copyright © 2024'
    }
  }
}) 