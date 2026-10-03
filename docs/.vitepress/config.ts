import { defineConfig } from 'vitepress';
import { withMermaid } from 'vitepress-plugin-mermaid';

export default withMermaid(
  defineConfig({
    title: 'Vigilo Stream',
    description:
      'Zero-copy multi-modal stream fusion engine for real-time AI pipelines in Python.',
    base: '/vigilo-stream/',
    lastUpdated: true,
    ignoreDeadLinks: false,
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: '/vigilo-stream/logo.svg' }],
      ['meta', { name: 'theme-color', content: '#f59e0b' }],
      ['meta', { property: 'og:title', content: 'Vigilo Stream Documentation' }],
      [
        'meta',
        {
          property: 'og:description',
          content: 'Zero-copy multi-modal stream fusion engine for real-time AI pipelines in Python.',
        },
      ],
      ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ],
    appearance: 'dark',
    themeConfig: {
      logo: '/logo.svg',
      darkModeSwitchLabel: 'Appearance',
      lightModeSwitchTitle: 'Switch to light theme',
      darkModeSwitchTitle: 'Switch to dark theme',
      nav: [
        { text: 'Guide', link: '/guide/getting-started' },
        { text: 'API Reference', link: '/api/pipeline' },
        { text: 'Architecture', link: '/architecture/overview' },
        { text: 'Hardware & GPU', link: '/hardware/acceleration' },
        {
          text: 'v1.0.2',
          items: [
            { text: 'PyPI Package', link: 'https://pypi.org/project/vigilo-stream/' },
            { text: 'GitHub Releases', link: 'https://github.com/Abdullah-Masood-05/vigilo-stream/releases' },
            { text: 'vigilo-core (Rust)', link: 'https://github.com/Abdullah-Masood-05/vigilo-core' },
          ],
        },
      ],
      sidebar: {
        '/guide/': [
          {
            text: 'Getting started',
            items: [
              { text: 'What is vigilo-stream?', link: '/guide/what-is-vigilo' },
              { text: 'Installation & quickstart', link: '/guide/getting-started' },
            ],
          },
          {
            text: 'Core concepts',
            items: [
              { text: 'Zero-copy memory', link: '/guide/zero-copy' },
              { text: 'Pipeline lifecycle', link: '/guide/pipeline' },
              { text: 'Fusion engine & replay', link: '/guide/fusion-engine' },
            ],
          },
          {
            text: 'Visualizations',
            items: [
              { text: 'Live OpenCV HUD', link: '/guide/opencv-hud' },
            ],
          },
        ],
        '/api/': [
          {
            text: 'Python API reference',
            items: [
              { text: 'Pipeline', link: '/api/pipeline' },
              { text: 'Frame & buffer sharing', link: '/api/frame' },
              { text: 'FusionEngine', link: '/api/fusion' },
              { text: 'Detection & signal types', link: '/api/types' },
              { text: 'Violations & events', link: '/api/events' },
              { text: 'Configuration schema', link: '/api/config' },
              { text: 'Model weights & helpers', link: '/api/models' },
            ],
          },
        ],
        '/architecture/': [
          {
            text: 'System architecture',
            items: [
              { text: 'Overview & thread topology', link: '/architecture/overview' },
            ],
          },
        ],
        '/hardware/': [
          {
            text: 'Hardware acceleration',
            items: [
              { text: 'CPU vs discrete GPU', link: '/hardware/acceleration' },
            ],
          },
        ],
      },
      socialLinks: [
        { icon: 'github', link: 'https://github.com/Abdullah-Masood-05/vigilo-stream' },
      ],
      footer: {
        message: 'Released under the AGPL-3.0 License.',
        copyright: 'Copyright © 2026 Abdullah Masood',
      },
      search: { provider: 'local' },
    },
    mermaid: {
      theme: 'base',
      themeVariables: {
        darkMode: true,
        background: '#0f1117',
        primaryColor: '#1a1e29',
        primaryBorderColor: '#363d4f',
        primaryTextColor: '#f3f4f6',
        secondaryColor: '#242a38',
        tertiaryColor: '#12151e',
        lineColor: '#9ca3af',
        textColor: '#d1d5db',
        clusterBkg: '#12151e',
        clusterBorder: '#2d3444',
        edgeLabelBackground: '#1a1e29',
        nodeBorder: '#f59e0b',
        mainBkg: '#1a1e29',
      },
    },
  }),
);
