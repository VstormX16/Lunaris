import { Component, type ReactNode } from 'react'

type Props = { children: ReactNode; /** what to show instead when the children crash */ fallback: (error: Error, reset: () => void) => ReactNode }

/** Keeps a crash in one part of the screen from taking the whole app with it (React unmounts everything on an uncaught render error). */
export default class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error) {
    console.error('Caught by ErrorBoundary:', error)
  }

  render() {
    return this.state.error ? this.props.fallback(this.state.error, () => this.setState({ error: null })) : this.props.children
  }
}
